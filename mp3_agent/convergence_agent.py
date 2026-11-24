"""typed global selection, one audit check, finite raw-root rescues."""
from collections import deque
from mp3_agent.audit_agent import AuditAgent
from mp3_agent.audit_facts import weak_feedback
from mp3_agent.convergence_facts import ConvergenceFacts
from mp3_agent.convergence_rescue import proposals
from mp3_agent.rule_support import text
from mp3_agent.safety import sql_key,sql_within_limit


class ConvergenceAgent(AuditAgent):
    def __init__(self,config,log=None):
        super().__init__(config,log)
        self.convergence=config['s23_enabled'];self.rescue_done=False
        self.rescue_events=[];self.first_hint_done=False;self.priority_audit_used=False

    def diagnosis(self,row):
        d=super().diagnosis(row)
        if self.convergence and isinstance(self.facts,ConvergenceFacts) and 'unconnected' not in d:
            d['unconnected']=self.facts.unconnected(self.tree(row))
            d['missing_count_scope']=self.facts.missing_count_scope(self.tree(row))
        return d

    def origin_rank(self,row):
        return min((self.config['source_tiers'][o['kind']],o['steps'],o['root_order'],row['key']) for o in row['origins'])

    def selection_rank(self,row):
        d=self.diagnosis(row)
        return (sum(e['state']=='contradicted' for e in d['output']),
                sum(e['state']=='contradicted' for e in d['tasks']),len(d['unconnected']),len(d['risks']),
                d['missing_count_scope'],-sum(e['state']=='satisfied' for e in d['tasks']),*self.origin_rank(row))

    def priority(self,row):
        if not self.convergence or not isinstance(self.facts,ConvergenceFacts):return super().priority(row)
        known_bad=self.diagnosis(row)['schema_state'] is False
        if row.get('feedback',{} ) and row['feedback'].get('ok'):known_bad=False
        return (int(known_bad),*self.selection_rank(row))

    def group_rank(self,rows):
        # A coherent member/origin tuple; never splice origins across a group.
        return min(self.selection_rank(r) for r in rows)

    def select(self,groups):
        passed=[(g,rows,self.actual(rows)) for g,rows in groups.items() if self.actual(rows) and self.actual(rows)['feedback'].get('matches_example')]
        executable=[(g,rows,self.actual(rows)) for g,rows in groups.items() if self.actual(rows) and self.actual(rows)['feedback'].get('ok')]
        unchecked=[(g,rows,min(rows,key=self.priority)) for g,rows in groups.items() if self.actual(rows) is None]
        pool=passed or executable or unchecked
        return min(pool,key=lambda p:self.group_rank(p[1]))[2] if pool else None

    def replacement(self,old,new):
        if not self.convergence:return super().replacement(old,new)
        return self.selection_rank(new)<self.selection_rank(old),'global_lexicographic_rank'

    def rescue_needed(self):
        groups=self.groups();chosen=self.select(groups)
        if not chosen or not chosen.get('feedback') or not chosen['feedback'].get('matches_example'):return True
        d=self.diagnosis(chosen)
        needs={'count_entity','count_distinct_identity','entity_group_count','entity_name_binding'}
        return bool(weak_feedback(chosen['feedback']) or any(e['state']=='contradicted' for e in d['output']+d['tasks'])
                    or d['unconnected'] or d['risks'] or any(e['id'] in needs and e['state']!='satisfied' for e in d['tasks']))

    def generate(self):
        if not self.convergence:return super().generate()
        if not self.rescue_done:
            self.rescue_done=True
            if self.rescue_needed():
                roots=sorted((r for r in self.candidates.values() if any(o['kind'] in ('hint','original') for o in r['origins'])),
                             key=lambda r:(min(o['root_order'] for o in r['origins'] if o['kind'] in ('hint','original')),r['key']))
                # Alternate families/root edits; each generator is finite and one level.
                family={}
                for parent,name,query,evidence in proposals(roots,self.facts):family.setdefault(name,deque()).append((parent,query,evidence))
                queue=deque(family)
                while queue and len(self.skeleton_keys)<self.config['max_skeleton_candidates']:
                    name=queue.popleft();parent,query,evidence=family[name].popleft()
                    sql=text(query);key=sql_key(sql);fresh=key not in self.candidates
                    raw=min((o for o in parent['origins'] if o['kind'] in ('original','hint')),key=lambda o:(o['root_order'],o['tie']))
                    origin={'kind':'rewrite','parent_key':parent['key'],'root_sql':parent['sql'],'root_order':raw['root_order'],
                            'steps':1,'tie':[2,raw['root_order'],7], 'skeleton_route':'rescue','skeleton_depth':1,
                            'rescue_family':name,'rescue_depth':1,'trigger':evidence,'path':[{'rule':name,'evidence':evidence}]}
                    if self.facts.schema_state(self.tree_for_sql(sql))[0] is not False:
                        child=self.add(sql,origin)
                        if child and fresh:self.skeleton_keys.add(key)
                        self.rescue_events.append({'family':name,'parent':parent['key'],'key':key,'fresh':fresh,'trigger':evidence})
                    if family[name]:queue.append(name)
        return super().generate()

    def expand_audits(self,groups):
        if not self.convergence:return super().expand_audits(groups)
        # Pure rescue lineage has no audit descendants. Mixed old lineage remains eligible.
        filtered={g:[{**r,'origins':[o for o in r['origins'] if not o.get('rescue_family')]} for r in rows
                     if any(not o.get('rescue_family') for o in r['origins'])] for g,rows in groups.items()}
        filtered={g:rows for g,rows in filtered.items() if rows}
        # Parent audit machinery already handles relation risks. Add disconnected source
        # evidence temporarily only as a generation trigger, never as an FK ranking risk.
        changed=[]
        for rows in filtered.values():
            for r in rows:
                d=self.diagnosis(r)
                if d['unconnected'] and not d['risks']:
                    changed.append(d);d['risks']=[{'generation_only':'unconnected_relation'}]
        try:return super().expand_audits(filtered)
        finally:
            for d in changed:d['risks']=[]

    def eligible_audit(self,rows):
        return any(o.get('audit_parent') or (o.get('rescue_family') and o.get('trigger')) for r in rows for o in r['origins'])

    def finish(self,observation,reason):
        if not self.convergence:return super().finish(observation,reason)
        groups=self.groups();chosen=self.select(groups);log=self.log;self.log=None
        try:action=super().finish(observation,reason)
        finally:self.log=log
        if chosen:
            action={'tool':'submit','sql':chosen.get('checked_sql',chosen['sql'])}
            self.record['selection'].update(key=chosen['key'],sql=action['sql'])
            self.record['trace'][-1]=action;self.record['audit']['selected']=chosen['key']
        self.record['convergence']={'selection':'global_actual_public_group_minimum','priority_audit_used':self.priority_audit_used,
            'rescues':self.rescue_events,'group_ranks':{g:self.group_rank(rows) for g,rows in groups.items()}}
        if log:log.finish_task(self.record)
        return action

    def act(self,observation):
        if not self.convergence:return super().act(observation)
        self.acts+=1;self.allow_search=bool(observation['remaining']);self.ingest(observation)
        if self.schema is not None and self.facts is None:
            self.facts=ConvergenceFacts(observation['task']['question'],self.schema,observation['example'],self.config,observation['issue_sql'])
        history=observation['history']
        if not history:return {'tool':'schema'}
        if len(history)==1:
            return {'tool':'check','sql':observation['issue_sql']} if sql_within_limit(observation['issue_sql']) else self.finish(observation,'original_outside_tool_contract')
        if len(history)==2:return {'tool':'repair_hint'}
        if not observation['remaining']:return self.finish(observation,'tool_budget')
        groups=self.groups()
        if any(self.actual(rows) and self.actual(rows)['checked_at']>1 and any(o['kind']=='hint' for r in rows for o in r['origins']) for rows in groups.values()):
            self.first_hint_done=True
        if not self.first_hint_done:
            self.first_hint_done=True
            hints=[(g,r) for g,r in self.unchecked_groups(groups) if any(o['kind']=='hint' for x in groups[g] for o in x['origins']) and self.diagnosis(r)['schema_state'] is not False]
            if hints:
                group,row=min(hints,key=lambda p:self.priority(p[1]))
                return self.check_action(group,row,observation,'first_eligible_hint')
        self.generate();groups=self.groups();self.expand_audits(groups);groups=self.groups();unchecked=self.unchecked_groups(groups)
        # Any pending probe child is checked before spending another resource.
        if history[-1]['tool']=='execute':
            child_key=self.probes[-1]['result_key'] if self.probes else None
            pair=next(((g,r) for g,r in unchecked if any(x['key']==child_key for x in groups[g])),None)
            if pair:return self.check_action(pair[0],self.candidates[child_key],observation,'pending_probe_child')
        passed=any(self.actual(rows) and self.actual(rows)['feedback'].get('matches_example') for rows in groups.values())
        if observation['remaining']==self.config['tail_probe_remaining'] and not passed and not self.probes:
            options=self.probe_options(tail=True)
            if options:return self.start_probe(options[0],observation,'tail_probe_protection',unchecked)
        if any(self.actual(rows) and self.eligible_audit(rows) for rows in groups.values()):self.priority_audit_used=True
        if not self.priority_audit_used:
            eligible=[(g,r) for g,r in unchecked if self.eligible_audit(groups[g]) and self.diagnosis(r)['schema_state'] is not False]
            if eligible:
                group,row=min(eligible,key=lambda p:self.priority(p[1]));self.priority_audit_used=True
                return self.check_action(group,row,observation,'priority_audit')
        if unchecked:
            group,row=min(unchecked,key=lambda p:self.priority(p[1]));return self.check_action(group,row,observation,'best_unchecked_group')
        if observation['remaining']>=2:
            options=self.probe_options()
            if options:return self.start_probe(options[0],observation,'no_checkable_candidate',unchecked)
        return self.finish(observation,'candidates_and_diagnostics_exhausted')

    def check_action(self,group,row,observation,reason):
        self.decision_events.append({'history_length':len(observation['history']),'remaining':observation['remaining'],
            'tool':'check','key':row['key'],'group':group,'priority':self.priority(row),'reason':reason})
        return {'tool':'check','sql':row['sql']}
