"""combined full-budget policy. Inputs are current official observations only."""
from collections import deque
import json
import time
from sqlglot import exp
from mp3_sql.sqlite_tool import validate_sql
from mp3_agent.skeleton_agent import SkeletonAgent
from mp3_agent.skeleton_rebuild import proposals
from mp3_agent.semantic_facts import SemanticFacts
from mp3_agent.audit_facts import AuditFacts, weak_feedback
from mp3_agent.execution_identity import identity, physical_expression, bound_query
from mp3_agent.rule_support import parse, text, unalias
from mp3_agent.rule_rewrites import probe_requests
from mp3_agent.rule_controller import RuleAgent
from mp3_agent.safety import sql_key, sql_within_limit


class AuditAgent(SkeletonAgent):
    def __init__(self, config, log=None):
        super().__init__(config, log)
        self.audit_enabled=config['audit_enabled']; self.audit_done=set(); self.audit_events=[]
        self.identity_cache={}; self.diagnosis_cache={}; self.group_events=[]
        self.replacement_events=[]; self.tail_events=[]; self.probe_child=None
        self.allow_search=True

    def expand(self):
        if self.audit_enabled and not self.allow_search:return
        return super().expand()

    def diagnosis(self,row):
        if row['key'] not in self.diagnosis_cache:
            tree=self.tree(row)
            state,bad=self.facts.schema_state(tree)
            self.diagnosis_cache[row['key']]={'schema_state':state,'bad_columns':bad,
                'output':self.facts.output_vector(tree),'tasks':self.facts.evaluate(tree),
                'risks':self.facts.relation_risks(tree),'count':self.facts.count_identity(tree)}
        return self.diagnosis_cache[row['key']]

    def evidence(self,row):
        if not self.audit_enabled or not isinstance(self.facts,AuditFacts): return super().evidence(row)
        d=self.diagnosis(row)
        return d['output']+d['tasks']

    def vector(self,row):
        if not self.audit_enabled or not isinstance(self.facts,AuditFacts): return super().vector(row)
        return self.diagnosis(row)['tasks']

    def priority(self,row):
        if not self.audit_enabled or not isinstance(self.facts,AuditFacts): return super().priority(row)
        d=self.diagnosis(row);o=self.best_origin(row)
        return (int(d['schema_state'] is False),sum(e['state']=='contradicted' for e in d['output']),
                sum(e['state']=='contradicted' for e in d['tasks']),len(d['risks']),
                -sum(e['state']=='satisfied' for e in d['tasks']),o['steps'],tuple(o['tie']),row['key'])

    def rank(self,row):
        return self.priority(row) if self.audit_enabled and isinstance(self.facts,AuditFacts) else super().rank(row)

    def variants(self,row,family):
        for sql,rid,guard,detail in super().variants(row,family):
            if self.audit_enabled and rid in {25,122,172,181}:
                keep=self.facts.preserves_groups(parse(row['sql']),parse(sql))
                if keep is not True:
                    self.guard_events.append({'parent_key':row['key'],'rule_id':rid,'state':keep,
                                              'reason':'group_dimension_not_preserved','rejected_sql':sql})
                    continue
            yield sql,rid,guard,detail

    def groups(self):
        groups={}
        for row in self.candidates.values():
            if row['key'] not in self.identity_cache:
                self.identity_cache[row['key']]=identity(row['sql'],self.schema)
            canonical,reason=self.identity_cache[row['key']]
            key='bound:'+canonical if canonical is not None else 'ast:'+row['key']
            groups.setdefault(key,[]).append(row)
        result={}
        for key,rows in groups.items():
            def signature(r):
                d=self.diagnosis(r)
                return (d['schema_state'],tuple((e['id'],e['state']) for e in d['output']+d['tasks']),
                        tuple(sorted(tuple(x['pair']) for x in d['risks'])))
            if len({signature(r) for r in rows})>1:
                for row in rows:result['ast:'+row['key']]=[row]
                event={'reason':'diagnosis_disagreement_split','members':[r['key'] for r in rows]}
                if event not in self.group_events:self.group_events.append(event)
            else:result[key]=rows
        return result

    @staticmethod
    def actual(rows):
        checked=[r for r in rows if r['feedback'] is not None]
        return min(checked,key=lambda r:r['checked_at']) if checked else None

    def unchecked_groups(self,groups):
        return [(g,min(rows,key=self.priority)) for g,rows in groups.items() if self.actual(rows) is None]

    def expand_audits(self,groups):
        queue=deque();started=time.perf_counter()
        for group,rows in groups.items():
            parent=self.actual(rows)
            if not parent or group in self.audit_done:continue
            # A mixed ordinary lineage retains eligibility; pure audit descendants never recurse.
            if not any(not o.get('audit_parent') for o in parent['origins']):continue
            f=parent['feedback'];d=self.diagnosis(parent)
            routes=set();reasons=[]
            if f.get('ok') and self.facts.count_table and (d['count']['state']!='satisfied' or
                    any(e['id']=='count_distinct_identity' and e['state']!='satisfied' for e in d['tasks'])):
                routes.update(('S','J'));reasons.append('count_identity_unresolved')
            if f.get('ok') and d['risks']:routes.add('J');reasons.append('relation_risk')
            if weak_feedback(f):routes.update(('S','J'));reasons.append('weak_public_example')
            if not routes:continue
            self.audit_done.add(group)
            for route in ('S','J'):
                if route not in routes:continue
                event={'group':group,'parent':parent['key'],'route':route,'reasons':reasons,'events':[],'added':[]}
                self.audit_events.append(event)
                # Empty labels disable target-header plans; still filter plan explicitly below.
                gen=proposals(parent['sql'],[],self.schema,self.facts.question,self.facts.count_table,self.config,route,event['events'])
                queue.append((parent,route,event,gen))
        while queue and len(self.skeleton_keys)<self.config['max_skeleton_candidates']:
            parent,route,event,gen=queue.popleft()
            for item in gen:
                if item['projection_plan']!='source':continue
                po=min((o for o in parent['origins'] if not o.get('audit_parent')),key=lambda o:(o['steps'],o['tie']))
                origin={'kind':'rewrite','parent_key':parent['key'],'root_sql':parent['sql'],
                    'root_order':po['root_order'],'steps':po['steps']+1,'tie':[2,po['root_order'],6,('S','J').index(route)],
                    'skeleton_route':route,'skeleton_depth':po.get('skeleton_depth',0)+1,
                    'audit_parent':parent['key'],'audit_depth':1,
                    'path':po.get('path',[])+[{'rule':'feedback_audit','evidence':item,'risks':item['risks']}]}
                try:key=sql_key(item['sql'])
                except ValueError:continue
                fresh=key not in self.candidates
                if self.facts.schema_state(self.tree_for_sql(item['sql']))[0] is False:continue
                child=self.add(item['sql'],origin)
                if child and fresh:
                    self.skeleton_keys.add(child['key']);event['added'].append(child['key']);queue.append((parent,route,event,gen));break
        if queue:self.audit_events.append({'reason':'shared_skeleton_cap','pending_groups':len(queue)})
        self.skeleton_seconds+=time.perf_counter()-started

    def probe_options(self,tail=False):
        options=[]
        for row in self.diagnostic_rows():
            if self.facts.schema_state(self.tree(row))[0] is False:continue
            if tail and (not row['feedback'] or not row['feedback'].get('ok') or row['feedback'].get('matches_example')):continue
            for request in probe_requests(row,self.constraints,self.schema):
                if tuple(request['identity']) not in self.probed:options.append((self.priority(row),request))
        return sorted(options,key=lambda x:(x[0],x[1]['identity']))

    def start_probe(self,option,observation,reason,unchecked):
        _,request=option;self.pending_probe=request;self.probed.add(tuple(request['identity']))
        event={'history_length':len(observation['history']),'remaining':observation['remaining'],
               'tool':'execute','identity':request['identity'],'reason':reason,
               'deferred_groups':[g for g,_ in sorted(unchecked,key=lambda p:self.priority(p[1]))]}
        self.decision_events.append(event)
        if reason=='tail_probe_protection':self.tail_events.append(event)
        return {'tool':'execute','sql':request['sql']}

    def risk_preserved(self,old,new):
        """Positive relation repair cannot silently change filters or ordinary outputs."""
        a=self.tree(old);b=self.tree(new)
        if not isinstance(a,exp.Select) or not isinstance(b,exp.Select):return False
        def expr(node,root):
            return physical_expression(node,root,self.schema)
        for k in ('where','group','having','order','limit','offset'):
            x=a.args.get(k);y=b.args.get(k)
            if (x is None)!=(y is None):return False
            if x is not None:
                xx=expr(x,a);yy=expr(y,b)
                if xx is None or yy is None or xx!=yy:return False
        if bool(a.args.get('distinct'))!=bool(b.args.get('distinct')) or len(a.expressions)!=len(b.expressions):return False
        for x,y in zip(a.expressions,b.expressions):
            x=unalias(x);y=unalias(y)
            xx=expr(x,a);yy=expr(y,b)
            if xx is not None and xx==yy:continue
            ca=self.facts.count_identity(a);cb=self.facts.count_identity(b)
            if isinstance(x,exp.Count) and isinstance(y,exp.Count) and ca['state']==cb['state']=='satisfied' and ca['identity']==cb['identity'] and ca['distinct']==cb['distinct']:
                continue
            return False
        # Replacement needs an actual FK connection between the same original endpoints.
        try:
            q,bindings,reason=bound_query(text(b),self.schema)
            if reason:return False
            graph={}
            for j in q.args.get('joins',[]):
                if j.args.get('side') or j.args.get('kind') not in (None,'INNER'):return False
                on=j.args.get('on')
                parts=list(on.flatten()) if isinstance(on,exp.And) else [on]
                for n in parts:
                    if not isinstance(n,exp.EQ) or not isinstance(n.this,exp.Column) or not isinstance(n.expression,exp.Column):return False
                    t,c=bindings[id(n.this)];u,d=bindings[id(n.expression)]
                    if (t,c,u,d) not in self.schema.edges.get(t,[]):return False
                    graph.setdefault(t,set()).add(u);graph.setdefault(u,set()).add(t)
            for risk in self.diagnosis(old)['risks']:
                start,end=risk['endpoints'];pending=[start];seen=set()
                while pending:
                    n=pending.pop()
                    if n in seen:continue
                    seen.add(n);pending.extend(graph.get(n,set())-seen)
                if end not in seen:return False
            return True
        except (ValueError,KeyError,TypeError):return False

    def replacement(self,old,new):
        a=self.diagnosis(old);b=self.diagnosis(new)
        before={e['id']:e['state'] for e in a['output']+a['tasks']}
        after={e['id']:e['state'] for e in b['output']+b['tasks']}
        if set(before)!=set(after):return False,'different_obligations'
        if any((v=='satisfied' and after[k]!='satisfied') or (v!='contradicted' and after[k]=='contradicted') for k,v in before.items()):
            return False,'lost_support_or_new_conflict'
        gain=any(v!='satisfied' and after[k]=='satisfied' for k,v in before.items() if not k.startswith('output_'))
        risk_gain=len(b['risks'])<len(a['risks']) and self.risk_preserved(old,new)
        if not gain and not risk_gain:return False,'no_positive_improvement'
        if self.priority(new)>=self.priority(old):return False,'key_not_better'
        return True,'supported_task_gain' if gain else 'evidenced_relation_repair'

    def finish(self,observation,reason):
        if not self.audit_enabled:return super().finish(observation,reason)
        groups=self.groups() if self.facts else {}
        actual=[self.actual(rows) for rows in groups.values()];actual=[r for r in actual if r]
        passed=sorted([r for r in actual if r['feedback'].get('matches_example')],key=lambda r:r['checked_at'])
        incumbent=passed[0] if passed else None
        for challenger in passed[1:]:
            replace,why=self.replacement(incumbent,challenger)
            self.replacement_events.append({'incumbent':incumbent['key'],'challenger':challenger['key'],'replace':replace,'reason':why})
            if replace:incumbent=challenger
        chosen=incumbent
        if chosen is None:
            executable=[r for r in actual if r['feedback'].get('ok')]
            unchecked=[r for _,r in self.unchecked_groups(groups)]
            chosen=min(executable or unchecked,key=self.priority,default=None)
        # Reuse the common record builder, avoiding the parent selection overrides.
        log=self.log;self.log=None
        try:action=RuleAgent.finish(self,observation,reason)
        finally:self.log=log
        if chosen:
            sql=chosen.get('checked_sql',chosen['sql'])
            action={'tool':'submit','sql':sql}
            self.record['selection'].update(key=chosen['key'],sql=sql)
            self.record['trace'][-1]=action
        sql_key(action['sql']);validate_sql(action['sql'])
        group_rows=[]
        for key,rows in groups.items():
            checked=self.actual(rows)
            group_rows.append({'identity':key,'members':[r['key'] for r in rows],
                'checked_key':checked['key'] if checked else None,
                'checked_sql':checked.get('checked_sql') if checked else None,
                'associations':[{'key':r['key'],'equivalent_to_checked_sql':checked.get('checked_sql') if checked and r is not checked else None} for r in rows]})
        self.record.update(semantic_obligations=self.facts.obligations,semantic_vectors={r['key']:self.vector(r) for r in self.candidates.values()},
            semantic_guards=self.guard_events,s1_counts={'new_ast':len(self.s1_keys),'guard_checks':len(self.guard_events),'max_depth':max((self.depth(r) for r in self.candidates.values()),default=0)},
            s1_limits=self.limit_events,s1_timing={'generation_seconds':self.generation_seconds},s1_decisions=self.decision_events,
            skeleton={'enabled':True,'anchors':self.skeleton_anchors,'new_keys':sorted(self.skeleton_keys),'events':self.skeleton_events,
                      'generation_seconds':self.skeleton_seconds,'max_depth':max((o.get('skeleton_depth',0) for r in self.candidates.values() for o in r['origins']),default=0)},
            audit={'diagnoses':{r['key']:self.diagnosis(r) for r in self.candidates.values()},'execution_groups':group_rows,
                   'identity_fallbacks':{k:v[1] for k,v in self.identity_cache.items() if v[1]},'group_events':self.group_events,
                   'generation':self.audit_events,'replacements':self.replacement_events,'tail_probes':self.tail_events,
                   'selected':chosen['key'] if chosen else None,'initial_incumbent':passed[0]['key'] if passed else None})
        self.record['configuration'].update(batch_id=self.config['batch_id'],audit_enabled=True,skeleton_enabled=True,s1_enabled=True)
        if log:log.finish_task(self.record)
        return action

    def act(self,observation):
        if not self.audit_enabled:return super().act(observation)
        self.acts+=1;self.allow_search=bool(observation['remaining']);self.ingest(observation)
        if self.schema is not None and self.facts is None:
            self.facts=AuditFacts(observation['task']['question'],self.schema,observation['example'],self.config)
        history=observation['history']
        if not history:return {'tool':'schema'}
        if len(history)==1:
            return {'tool':'check','sql':observation['issue_sql']} if sql_within_limit(observation['issue_sql']) else self.finish(observation,'original_outside_tool_contract')
        if len(history)==2:return {'tool':'repair_hint'}
        if not observation['remaining']:return self.finish(observation,'tool_budget')
        self.generate();groups=self.groups();self.expand_audits(groups);groups=self.groups()
        unchecked=self.unchecked_groups(groups)
        # A valid child of the immediately preceding tail probe gets the last check.
        if history[-1]['tool']=='execute' and self.tail_events and self.tail_events[-1]['history_length']==len(history)-1:
            child_key=self.probes[-1]['result_key'] if self.probes else None
            pair=next(((g,r) for g,r in unchecked if any(x['key']==child_key for x in groups[g])),None)
            if pair:
                child=self.candidates[child_key]
                self.decision_events.append({'tool':'check','key':child_key,'remaining':observation['remaining'],'reason':'tail_probe_child'})
                return {'tool':'check','sql':child['sql']}
        passed=any((r:=self.actual(rows)) and r['feedback'].get('matches_example') for rows in groups.values())
        if observation['remaining']==self.config['tail_probe_remaining'] and not passed and not self.probes:
            options=self.probe_options(tail=True)
            if options:return self.start_probe(options[0],observation,'tail_probe_protection',unchecked)
        if unchecked:
            group,row=min(unchecked,key=lambda p:self.priority(p[1]))
            self.decision_events.append({'history_length':len(history),'remaining':observation['remaining'],
                'tool':'check','key':row['key'],'group':group,'priority':self.priority(row),'reason':'best_unchecked_group'})
            return {'tool':'check','sql':row['sql']}
        if observation['remaining']>=2:
            options=self.probe_options()
            if options:return self.start_probe(options[0],observation,'no_checkable_candidate',unchecked)
        return self.finish(observation,'candidates_and_diagnostics_exhausted')
