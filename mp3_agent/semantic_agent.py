"""2026-11-21 12:53 HKT. Approved causal search, separate from frozen V."""
from collections import deque
import json
import time

from mp3_agent.rule_controller import RuleAgent
from mp3_agent.rule_rewrites import probe_requests
from mp3_agent.rule_support import parse, text
from mp3_agent.safety import sql_key, sql_within_limit
from mp3_agent.semantic_ops import RULE_IDS, FAMILIES, normalize, edit
from mp3_agent.semantic_facts import SemanticFacts, improves


class SemanticAgent(RuleAgent):
    def __init__(self,config,log=None):
        super().__init__(log,rules_enabled=True,run_context={k:config[k] for k in ('condition','purpose')})
        self.config=config;self.enabled=config['s1_enabled'];self.facts=None
        self.semantic_cache={};self.tree_cache={};self.guard_events=[];self.s1_expanded=set()
        self.s1_keys=set();self.limit_events=[];self.selection_detail=None
        self.generation_seconds=0.;self.decision_events=[]

    @staticmethod
    def legacy(row):
        return [o for o in row['origins'] if not o.get('s1_depth',0)]

    def expand(self):
        if not self.enabled:return super().expand()
        all_rows=self.candidates
        legacy={k:{**r,'origins':self.legacy(r)} for k,r in all_rows.items() if self.legacy(r)}
        self.candidates=legacy
        try:super().expand()
        finally:
            for key,row in self.candidates.items():
                if key not in all_rows:all_rows[key]=row
                else:
                    for origin in row['origins']:
                        if origin not in all_rows[key]['origins']:all_rows[key]['origins'].append(origin)
            self.candidates=all_rows

    def add_rewrite(self,parent,item):
        if not self.enabled or self.legacy(parent):return super().add_rewrite(parent,item)
        return self.add_semantic(parent,item['sql'],None,{}, {'legacy_probe':item},'literal_format')

    def origin(self,row):
        return min(row['origins'],key=lambda o:(o.get('s1_depth',0),o['steps'],o['tie']))

    def depth(self,row):return self.origin(row).get('s1_depth',0)

    def tree(self,row):
        if row['key'] not in self.tree_cache:self.tree_cache[row['key']]=normalize(parse(row['sql']))
        return self.tree_cache[row['key']]

    def vector(self,row):
        if row['key'] not in self.semantic_cache:self.semantic_cache[row['key']]=self.facts.evaluate(self.tree(row))
        return self.semantic_cache[row['key']]

    def priority(self,row):
        v=self.vector(row);violations=sum(e['state']=='contradicted' for e in v);sat=sum(e['state']=='satisfied' for e in v)
        schema_state,_=self.facts.schema_state(self.tree(row))
        origins=[o for o in row['origins'] if o.get('s1_depth')]
        binding=any(o.get('rule_id')==106 for o in origins)
        level=0 if binding else 1 if origins and any(o.get('repairs_contradiction') for o in origins) else 2
        return (int(schema_state is False),level,violations,-sat,self.rank(row),self.depth(row),row['key'])

    def add_semantic(self,parent,sql,rid,guard,detail,family):
        origin=self.origin(parent);depth=origin.get('s1_depth',0)+1
        if depth>self.config['max_s1_depth']:
            self.limit_events.append({'reason':'s1_depth','parent_key':parent['key'],'rule_id':rid});return None
        try:key=sql_key(sql)
        except ValueError as e:
            self.rejections.append({'sql':sql,'reason':str(e),'s1_rule_id':rid});return None
        fresh=key not in self.candidates
        if fresh and len(self.s1_keys)>=self.config['max_s1_candidates']:
            self.limit_events.append({'reason':'s1_candidate_cap','parent_key':parent['key'],'rule_id':rid});return None
        tree=normalize(parse(sql));schema_state,bad=self.facts.schema_state(tree)
        if schema_state is False:
            self.rejections.append({'sql':sql,'reason':'s1_known_schema_error','columns':bad,'s1_rule_id':rid});return None
        parent_vector=self.vector(parent);child_vector=self.facts.evaluate(tree)
        origin_path=origin.get('path',[])+[{'rule':'s1_'+family,'evidence':{'rule_id':rid,'guard':guard,**detail},'risks':['guarded_semantic_repair']}]
        new_origin={'kind':'rewrite','parent_key':parent['key'],'root_sql':origin.get('root_sql',parent['sql']),
             'root_order':origin['root_order'],'steps':origin['steps']+1,'s1_depth':depth,'rule_id':rid,
             'repairs_contradiction':any(a['state']=='contradicted' and b['state']=='satisfied' for a,b in zip(parent_vector,child_vector)),
             'tie':[2,origin['root_order'],6,rid if rid is not None else 1000], 'path':origin_path}
        row=self.add(sql,new_origin)
        if row and fresh:self.s1_keys.add(row['key'])
        return row

    def variants(self,row,family):
        from sqlglot import exp
        tree=self.tree(row)
        focuses=[n for n in tree.walk() if isinstance(n,(exp.Select,exp.Union))]
        feedback=row.get('feedback')
        fingerprint=json.dumps({k:feedback.get(k) for k in ('ok','error','matches_example','truncated','row_count')} if feedback else None,sort_keys=True)
        for rid in RULE_IDS:
            if FAMILIES[rid]!=family:continue
            # A concrete SQL error only opens binding/correlation repair descendants.
            if feedback and not feedback.get('ok'):
                message=json.dumps(feedback.get('error',{})).casefold()
                allowed={106,132} if any(s in message for s in ('no such column','no such table','ambiguous column')) else {157} if 'misuse of aggregate' in message else set()
                if rid not in allowed:continue
            for index,focus in enumerate(focuses):
                identity=(row['key'],rid,index,fingerprint)
                if identity in self.s1_expanded:continue
                self.s1_expanded.add(identity)
                guard,roles=self.facts.context(tree,rid,focus)
                state=True if all(g['state'] is True for g in guard.values()) else False if any(g['state'] is False for g in guard.values()) else None
                event={'parent_key':row['key'],'rule_id':rid,'scope_index':index,'feedback_fingerprint':fingerprint,
                       'guard':guard,'state':state,'generated':0}
                self.guard_events.append(event)
                if state is not True:continue
                for changed in edit(tree,rid,roles,focus):
                    event['generated']+=1
                    yield text(changed),rid,guard,{'scope_index':index,'feedback_fingerprint':fingerprint}

    def generate(self):
        started=time.perf_counter()
        try:return self._generate()
        finally:self.generation_seconds+=time.perf_counter()-started

    def _generate(self):
        # Fair rounds per anchor and family. Newly reached depth-one anchors form the next frontier.
        while True:
            if len(self.s1_keys)>=self.config['max_s1_candidates']:
                if not any(e['reason']=='s1_candidate_cap' for e in self.limit_events):self.limit_events.append({'reason':'s1_candidate_cap'})
                break
            before=len(self.s1_keys)
            groups=[]
            for row in sorted(list(self.candidates.values()),key=self.priority):
                if self.depth(row)>=self.config['max_s1_depth']:continue
                for family in sorted(set(FAMILIES.values())):
                    groups.append((row,self.variants(row,family),family))
            queue=deque(groups)
            while queue and len(self.s1_keys)<self.config['max_s1_candidates']:
                row,generator,family=queue.popleft()
                try:
                    sql,rid,guard,detail=next(generator)
                    self.add_semantic(row,sql,rid,guard,detail,family)
                    queue.append((row,generator,family))
                except StopIteration:pass
            if len(self.s1_keys)==before:break

    def finish(self,observation,reason):
        if not self.enabled:return super().finish(observation,reason)
        started=time.perf_counter()
        passed=[r for r in self.candidates.values() if r['feedback'] and r['feedback'].get('matches_example')]
        incumbent=min((r for r in passed if self.legacy(r)),key=self.rank,default=None)
        eligible=[r for r in passed if incumbent is None or improves(self.vector(incumbent),self.vector(r))]
        def final_rank(r):
            v=self.vector(r);o=self.origin(r)
            return (sum(e['state']=='contradicted' for e in v),-sum(e['state']=='satisfied' for e in v),
                    self.depth(r),o['steps'],tuple(o['tie']),r['key'])
        chosen=min(eligible,key=final_rank) if eligible else incumbent
        log=self.log;self.log=None
        try:action=super().finish(observation,reason)
        finally:self.log=log
        record=self.record
        if chosen:
            action={'tool':'submit','sql':chosen['sql']}
            record['selection'].update(key=chosen['key'],sql=chosen['sql'])
            record['trace'][-1]=action
        self.selection_detail={'incumbent':incumbent['key'] if incumbent else None,
            'incumbent_vector':self.vector(incumbent) if incumbent else None,
            'eligible':[r['key'] for r in eligible], 'selected':record['selection']['key']}
        record['configuration']['s1_enabled']=True
        record['semantic_obligations']=self.facts.obligations
        record['semantic_vectors']={r['key']:self.vector(r) for r in self.candidates.values()}
        record['semantic_guards']=self.guard_events;record['semantic_selection']=self.selection_detail
        record['s1_counts']={'new_ast':len(self.s1_keys),'guard_checks':len(self.guard_events),
            'max_depth':max((self.depth(r) for r in self.candidates.values()),default=0)}
        record['s1_limits']=self.limit_events
        record['s1_timing']={'generation_seconds':self.generation_seconds,
                             'selection_seconds':time.perf_counter()-started}
        record['s1_decisions']=self.decision_events
        if log:log.finish_task(record)
        return action

    def act(self,observation):
        if not self.enabled:return super().act(observation)
        self.acts+=1;self.ingest(observation)
        if self.schema is not None and self.facts is None:self.facts=SemanticFacts(observation['task']['question'],self.schema,observation['example'])
        history=observation['history']
        if not history:return {'tool':'schema'}
        if len(history)==1:
            return {'tool':'check','sql':observation['issue_sql']} if sql_within_limit(observation['issue_sql']) else self.finish(observation,'original_outside_tool_contract')
        if len(history)==2:return {'tool':'repair_hint'}
        if not observation['remaining']:return self.finish(observation,'tool_budget')
        self.generate()
        unchecked=[r for r in self.candidates.values() if r['feedback'] is None]
        if unchecked:
            chosen=min(unchecked,key=self.priority)
            self.decision_events.append({'history_length':len(history),'remaining':observation['remaining'],
                'tool':'check','key':chosen['key'],'priority':self.priority(chosen),'probe_deferred':'checkable_candidate'})
            return {'tool':'check','sql':chosen['sql']}
        diagnoses=[]
        if observation['remaining']>=2:
            for row in self.diagnostic_rows():
                state,bad=self.facts.schema_state(self.tree(row))
                if state is False:
                    continue
                for request in probe_requests(row,self.constraints,self.schema):
                    if tuple(request['identity']) not in self.probed:diagnoses.append((self.rank(row),request))
        if diagnoses:
            _,request=min(diagnoses,key=lambda x:(x[0],x[1]['identity']))
            self.pending_probe=request;self.probed.add(tuple(request['identity']))
            self.decision_events.append({'history_length':len(history),'remaining':observation['remaining'],
                'tool':'execute','identity':request['identity'],'reason':'no_checkable_candidate'})
            return {'tool':'execute','sql':request['sql']}
        self.decision_events.append({'history_length':len(history),'remaining':observation['remaining'],
            'tool':'submit','probe_deferred':'insufficient_budget' if observation['remaining']<2 else 'no_eligible_probe'})
        return self.finish(observation,'candidates_and_diagnostics_exhausted')

    def diagnostic_rows(self):
        """Extension point; the original diagnostic population is unchanged."""
        return self.candidates.values()
