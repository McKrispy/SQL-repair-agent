"""feedback ledger and a single bounded, shared reconstruction wheel."""
from collections import deque
import json
import time
from mp3_agent.convergence_agent import ConvergenceAgent
from mp3_agent.convergence_facts import ConvergenceFacts
from mp3_agent.semantic_agent import SemanticAgent
from mp3_agent.convergence_rescue import proposals as rescues
from mp3_agent.skeleton_rebuild import proposals as rebuilds
from mp3_agent.semantic_search_build import proposals as semantics
from mp3_agent.semantic_search_facts import SearchFacts
from mp3_agent.semantic_search_specs import feedback_class
from mp3_agent.audit_facts import weak_feedback
from mp3_agent.safety import sql_key,sql_within_limit
from mp3_agent.rule_support import text


def old_origins(row):return [o for o in row['origins'] if not o.get('semantic_family')]


class SearchAgent(ConvergenceAgent):
    def __init__(self,config,log=None):
        super().__init__(config,log)
        self.search_enabled=config['s31_enabled'];self.semantic_ready=False
        self.wheel={name:deque() for name in config['generation_order']};self.wheel_order=deque(config['generation_order'])
        self.task_ledger=set();self.search_events=[];self.goal_events=[];self.raw_registered=False
        self.priority_eligibility_checks=[]

    def baseline_only(self,operation):
        # The inherited closures already exclude skeletons; also exclude pure
        # new-family provenance even if a future representation lacks that flag.
        all_rows=self.candidates
        self.candidates={k:{**r,'origins':old_origins(r)} for k,r in all_rows.items() if old_origins(r)}
        try:return super().baseline_only(operation)
        finally:
            for k,r in self.candidates.items():
                if k not in all_rows:all_rows[k]=r
                else:
                    for o in r['origins']:
                        if o not in all_rows[k]['origins']:all_rows[k]['origins'].append(o)
            self.candidates=all_rows

    def diagnostic_rows(self):
        return (r for r in super().diagnostic_rows() if old_origins(r))

    def enable_semantics(self,observation):
        if self.semantic_ready:return
        raw=[observation['issue_sql']]
        for e in observation['history']:
            if e['tool']=='repair_hint':raw+=e['observation']
        self.facts=SearchFacts(observation['task']['question'],self.schema,observation['example'],self.config,observation['issue_sql'],raw)
        self.diagnosis_cache.clear();self.semantic_cache.clear();self.semantic_ready=True

    def semantic_needed(self):
        chosen=self.select(self.groups())
        if chosen is None or not chosen.get('feedback') or not chosen['feedback'].get('matches_example'):return True
        d=self.diagnosis(chosen);ids={g['id'] for g in self.facts.goals}
        return bool(weak_feedback(chosen['feedback']) or d['unconnected'] or d['risks'] or
                    any(e['state']=='contradicted' for e in d['output']+d['tasks']) or
                    any(e['id'] in ids and e['state']!='satisfied' for e in d['tasks']))

    def enqueue(self,name,parent,generator,metadata):
        self.wheel[name].append({'parent':parent,'generator':iter(generator),'metadata':metadata})

    def raw_rows(self):
        return sorted((r for r in self.candidates.values() if any(o['kind'] in ('hint','original') for o in r['origins'])),
                      key=lambda r:(min(o['root_order'] for o in r['origins'] if o['kind'] in ('hint','original')),r['key']))

    def register_raw(self):
        if self.raw_registered:return
        self.raw_registered=True;roots=self.raw_rows()
        if self.rescue_needed():
            for parent,name,q,evidence in rescues(roots,self.facts):
                family='old_quantity' if name=='entity_group_count' else 'old_name'
                self.enqueue(family,parent,[(text(q),{'rescue_family':name,'trigger':evidence})],{'mode':'rescue'})
        for parent in roots:
            self.skeleton_anchors.append({'key':parent['key'],'root_order':min(o['root_order'] for o in parent['origins'])})
            for route in ('P','S','J'):
                events=[];self.skeleton_events.append({'anchor':parent['key'],'route':route,'events':events})
                gen=rebuilds(parent['sql'],self.constraints.labels,self.schema,self.facts.question,self.facts.count_table,self.config,route,events)
                self.enqueue(route,parent,((i['sql'],i) for i in gen),{'mode':'skeleton','route':route})

    def register_feedback(self):
        needed=self.semantic_needed();groups=self.groups()
        bootstrap=[e for r in self.raw_rows() if r.get('feedback') for e in [{'key':r['key'],'class':feedback_class(r['feedback']),'checked_at':r['checked_at']}]]
        for parent in sorted(list(self.candidates.values()),key=lambda r:(min((o['root_order'] for o in r['origins']),default=0),r['checked_at'] if r['checked_at'] is not None else -1,r['key'])):
            ordinary=[o for o in parent['origins'] if not o.get('semantic_family') and not o.get('audit_parent') and o.get('skeleton_depth',0)<=1]
            raw=any(o['kind'] in ('original','hint') for o in ordinary)
            if not ordinary or not raw and parent.get('feedback') is None:continue
            cls=feedback_class(parent.get('feedback'))
            if needed:
                for goal in self.facts.goals:
                    task=(parent['key'],'root',goal['id'],goal['family'],json.dumps(goal,sort_keys=True),cls)
                    if task in self.task_ledger:continue
                    self.task_ledger.add(task);events=[]
                    trigger={'parent_feedback_class':cls,'parent_checked_at':parent['checked_at'],'bootstrap_context':bootstrap if cls=='unseen' else [],'goal_id':goal['id']}
                    self.goal_events.append({'task':list(task),'trigger':trigger,'events':events})
                    gen=semantics(parent,goal,self.facts.example,self.schema,self.config,events)
                    self.enqueue(goal['family'],parent,((text(q),e) for q,e in gen),{'mode':'semantic','goal':goal,'trigger':trigger,'parent_origin':min(ordinary,key=lambda o:(o['steps'],o['root_order'],o['tie']))})
        # Existing feedback S/J audits now enter the very same outer wheel.
        for group,rows in groups.items():
            parent=self.actual(rows)
            if parent is None or group in self.audit_done:continue
            ordinary=[o for o in parent['origins'] if not o.get('audit_parent') and not o.get('semantic_family') and not o.get('rescue_family')]
            if not ordinary:continue
            f=parent['feedback'];d=self.diagnosis(parent);routes=set();reasons=[]
            if f.get('ok') and self.facts.count_table and (d['count']['state']!='satisfied' or any(e['id']=='count_distinct_identity' and e['state']!='satisfied' for e in d['tasks'])):routes.update(('S','J'));reasons.append('count_identity_unresolved')
            if f.get('ok') and (d['risks'] or d['unconnected']):routes.add('J');reasons.append('relation_risk')
            if weak_feedback(f):routes.update(('S','J'));reasons.append('weak_public_example')
            if not routes:continue
            self.audit_done.add(group)
            for route in ('S','J'):
                if route not in routes:continue
                events=[];event={'group':group,'parent':parent['key'],'route':route,'reasons':reasons,'events':events,'added':[]};self.audit_events.append(event)
                gen=rebuilds(parent['sql'],[],self.schema,self.facts.question,self.facts.count_table,self.config,route,events)
                self.enqueue('audit'+route,parent,((i['sql'],i) for i in gen if i['projection_plan']=='source'),{'mode':'audit','route':route,'event':event,'parent_origin':min(ordinary,key=lambda o:(o['steps'],o['tie']))})

    def consume(self,name,entry,sql,evidence):
        parent=entry['parent'];meta=entry['metadata'];mode=meta['mode']
        po=meta.get('parent_origin') or min((o for o in parent['origins'] if o['kind'] in ('hint','original')),key=lambda o:(o['root_order'],o['tie']))
        depth=po.get('skeleton_depth',0)+1 if mode in ('audit','semantic') else 1
        origin={'kind':'rewrite','parent_key':parent['key'],'root_sql':parent['sql'],'root_order':po['root_order'],
                'steps':po['steps']+1 if mode in ('audit','semantic') else 1,'tie':[2,po['root_order'],6,self.config['generation_order'].index(name)],
                'skeleton_route':name,'skeleton_depth':depth,'path':po.get('path',[])+[{'rule':name,'evidence':evidence}]}
        if depth>self.config['max_skeleton_depth']+self.config['max_audit_depth']:return False
        if mode=='semantic':origin.update(semantic_family=name,semantic_depth=1,trigger=meta['trigger'],goal_id=meta['goal']['id'])
        elif mode=='audit':origin.update(audit_parent=parent['key'],audit_depth=1)
        elif mode=='rescue':origin.update(rescue_family=evidence['rescue_family'],rescue_depth=1,trigger=evidence['trigger'])
        try:key=sql_key(sql)
        except ValueError as error:self.rejections.append({'reason':str(error),'sql':sql,'origin':origin});return False
        if self.facts.schema_state(self.tree_for_sql(sql))[0] is False:
            self.rejections.append({'reason':'rebuild_known_schema_error','sql':sql,'origin':origin});return False
        fresh=key not in self.candidates;child=self.add(sql,origin)
        if child and fresh:self.skeleton_keys.add(key)
        self.search_events.append({'family':name,'parent':parent['key'],'key':key,'fresh':bool(child and fresh),'trigger':origin.get('trigger')})
        if mode=='rescue':self.rescue_events.append({'family':origin['rescue_family'],'parent':parent['key'],'key':key,'fresh':bool(child and fresh),'trigger':origin['trigger']})
        if mode=='audit' and child and fresh:meta['event']['added'].append(key)
        return bool(child and fresh)

    def generate(self):
        if not self.search_enabled or not self.semantic_ready:return super().generate()
        self.baseline_only(lambda:SemanticAgent.generate(self))
        self.register_raw();self.register_feedback();started=time.perf_counter()
        active=sum(bool(q) for q in self.wheel.values())
        while active and len(self.skeleton_keys)<self.config['max_skeleton_candidates']:
            name=self.wheel_order.popleft();self.wheel_order.append(name);queue=self.wheel[name]
            if not queue:continue
            entry=queue.popleft();fresh=False;exhausted=False
            while not fresh:
                try:sql,evidence=next(entry['generator'])
                except StopIteration:exhausted=True;break
                fresh=self.consume(name,entry,sql,evidence)
            if not exhausted:queue.append(entry)
            active=sum(bool(q) for q in self.wheel.values())
        if active:self.search_events.append({'reason':'shared_skeleton_cap','pending':{n:len(q) for n,q in self.wheel.items() if q}})
        self.skeleton_seconds+=time.perf_counter()-started

    def expand_audits(self,groups):
        if not self.search_enabled:return super().expand_audits(groups)
        # Registered by generate() and fairly consumed with every other family.

    def eligible_audit(self,rows):
        return super().eligible_audit(rows) or any(o.get('semantic_family') and o.get('trigger') for r in rows for o in r['origins'])

    def check_action(self,group,row,observation,reason):
        if not self.search_enabled:return super().check_action(group,row,observation,reason)
        groups=self.groups();eligible=self.eligible_audit(groups[group])
        # Causal qualification is snapshotted before the check, never inferred
        # later from an origin appended to an already checked execution group.
        if eligible and self.semantic_ready:
            self.priority_audit_used=True
            self.priority_eligibility_checks.append({'history_length':len(observation['history']),'group':group,'key':row['key'],'reason':reason})
        action=super().check_action(group,row,observation,reason)
        self.decision_events[-1].update(semantic_ready=self.semantic_ready,eligible_before_check=eligible,
            visible_queue=[{'group':g,'key':r['key'],'priority':self.priority(r)} for g,r in sorted(self.unchecked_groups(groups),key=lambda p:self.priority(p[1]))],
            pending_iterators={n:len(q) for n,q in self.wheel.items() if q})
        return action

    def finish(self,observation,reason):
        if not self.search_enabled:return super().finish(observation,reason)
        log=self.log;self.log=None
        try:action=super().finish(observation,reason)
        finally:self.log=log
        self.record['semantic_search']={'enabled':True,'definition':self.facts.definition if self.semantic_ready else None,
            'tasks':self.goal_events,'generation':self.search_events,'priority_qualification_checks':self.priority_eligibility_checks,
            'pending_iterators':{n:len(q) for n,q in self.wheel.items() if q},'shared_new_ast':len(self.skeleton_keys)}
        if log:log.finish_task(self.record)
        return action

    def act(self,observation):
        if not self.search_enabled:return super().act(observation)
        self.acts+=1;self.allow_search=bool(observation['remaining']);self.ingest(observation)
        if self.schema is not None and self.facts is None:self.facts=ConvergenceFacts(observation['task']['question'],self.schema,observation['example'],self.config,observation['issue_sql'])
        history=observation['history']
        if not history:return {'tool':'schema'}
        if len(history)==1:return {'tool':'check','sql':observation['issue_sql']} if sql_within_limit(observation['issue_sql']) else self.finish(observation,'original_outside_tool_contract')
        if len(history)==2:return {'tool':'repair_hint'}
        if not self.first_hint_done:
            self.first_hint_done=True;groups=self.groups()
            hints=[(g,r) for g,r in self.unchecked_groups(groups) if any(o['kind']=='hint' for x in groups[g] for o in x['origins']) and self.diagnosis(r)['schema_state'] is not False]
            if hints and observation['remaining']:
                group,row=min(hints,key=lambda p:self.priority(p[1]));return self.check_action(group,row,observation,'first_eligible_hint')
        self.enable_semantics(observation)
        if not observation['remaining']:return self.finish(observation,'tool_budget')
        self.generate();groups=self.groups();unchecked=self.unchecked_groups(groups)
        if history[-1]['tool']=='execute':
            child_key=self.probes[-1]['result_key'] if self.probes else None
            pair=next(((g,r) for g,r in unchecked if any(x['key']==child_key for x in groups[g])),None)
            if pair:return self.check_action(pair[0],self.candidates[child_key],observation,'pending_probe_child')
        passed=any(self.actual(rs) and self.actual(rs)['feedback'].get('matches_example') for rs in groups.values())
        if observation['remaining']==self.config['tail_probe_remaining'] and not passed and not self.probes:
            options=self.probe_options(tail=True)
            if options:return self.start_probe(options[0],observation,'tail_probe_protection',unchecked)
        if not self.priority_audit_used:
            eligible=[(g,r) for g,r in unchecked if self.eligible_audit(groups[g]) and self.diagnosis(r)['schema_state'] is not False]
            if eligible:
                group,row=min(eligible,key=lambda p:self.priority(p[1]));return self.check_action(group,row,observation,'priority_audit')
        if unchecked:
            group,row=min(unchecked,key=lambda p:self.priority(p[1]));return self.check_action(group,row,observation,'best_unchecked_group')
        if observation['remaining']>=2:
            options=self.probe_options()
            if options:return self.start_probe(options[0],observation,'no_checkable_candidate',unchecked)
        return self.finish(observation,'candidates_and_diagnostics_exhausted')
