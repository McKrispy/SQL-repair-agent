"""Shared evidence for all origins; syntax is support, never a grade."""
from sqlglot import exp
from mp3_agent.convergence_facts import ConvergenceFacts
from mp3_agent.semantic_search_specs import define,owner,sources,physical,pk
from mp3_agent.rule_support import text,unalias,header_expression,parse
from mp3_agent.skeleton_rebuild import _parts


class SearchFacts(ConvergenceFacts):
    def __init__(self,question,schema,example,config,original,raw):
        super().__init__(question,schema,example,config,original)
        self.definition=define(question,example,schema,raw,config)
        self.goals=self.definition['goals']
        replaced=set()
        for g in self.goals:
            if g['family']=='R1':replaced.add('absence')
            if g['family']=='R2':replaced.add('both')
            if g['family'] in ('R3','R4'):replaced.update(('grouping','count_boundary','entity_group_count','count_entity','count_distinct_identity'))
        self.replaced=replaced
        self.obligations=[o for o in self.obligations if o['id'] not in replaced]+[{'id':g['id'],'evidence':g} for g in self.goals]

    def owner(self,c,select):
        outer=select.find_ancestor(exp.Select)
        p=owner(c,select,self.schema,outer)
        return p[0] if p else None

    def output_signature(self,node):
        if not isinstance(node,(exp.Column,exp.AggFunc)):return None
        root=node.find_ancestor(exp.Select)
        return physical(node,root,self.schema) if root is not None else None

    def output_vector(self,tree):
        result=super().output_vector(tree)
        if not isinstance(tree,exp.Select):return result
        for i,h in enumerate(self.example['columns']):
            header=header_expression(h)
            if not isinstance(header,exp.Count) or i>=len(tree.expressions):continue
            e=unalias(tree.expressions[i])
            if isinstance(e,exp.Subquery) and isinstance(e.this,exp.Select) and len(e.this.expressions)==1 and isinstance(unalias(e.this.expressions[0]),exp.Count):result[i+1]['state']='satisfied'
        return result

    def evaluate(self,tree):
        # ConvergenceFacts iterates self.obligations; its unknown placeholders for
        # the new IDs are replaced once, rather than appended as extra votes.
        old=super().evaluate(tree)
        result=[e for e in old if e['id'] not in self.replaced and e['id'] not in {g['id'] for g in self.goals}]
        for goal in self.goals:result.append({'id':goal['id'],'state':self.goal_state(tree,goal),'evidence':goal})
        return result

    def root_match(self,tree,root):
        ss=sources(tree)
        if not ss or root['table'] not in ss.values() or not root.get('certain'):return False
        if root['kind']=='entity' and tree.args.get('distinct'):
            keys=[owner(unalias(e),tree,self.schema) for e in tree.expressions if isinstance(unalias(e),exp.Column)]
            if (root['table'],root['key']) not in keys:return False
        return True

    def linked(self,sub,outer,root,target):
        """Scoped declared FK graph plus an explicit identity/domain correlation."""
        local=sources(sub);out=sources(outer)
        if not local or not out or target not in local.values():return False
        root_aliases={a for a,t in out.items() if t==root['table']};graph={};correlated=set()
        cond=[j.args.get('on') for j in sub.args.get('joins',[])]+[sub.args['where'].this if sub.args.get('where') else None]
        for condition in cond:
            if condition is None:continue
            for p in _parts(condition):
                if not isinstance(p,(exp.EQ,exp.Is)) or not isinstance(p.this,exp.Column) or not isinstance(p.expression,exp.Column):continue
                a,b=p.this,p.expression;pa=owner(a,sub,self.schema,outer);pb=owner(b,sub,self.schema,outer)
                if pa is None or pb is None:continue
                is_domain=root['kind']=='domain' and isinstance(p,exp.Is) and pa==pb==(root['table'],root['key'])
                is_fk=(*pa,*pb) in self.schema.edges.get(pa[0],[]) and isinstance(p,exp.EQ)
                if not is_domain and not is_fk:continue
                aa=a.table.casefold();bb=b.table.casefold()
                graph.setdefault(aa,set()).add(bb);graph.setdefault(bb,set()).add(aa)
                if aa in root_aliases and aa not in local:correlated.add(aa)
                if bb in root_aliases and bb not in local:correlated.add(bb)
        pending=list(correlated);seen=set()
        while pending:
            a=pending.pop()
            if a in seen:continue
            seen.add(a);pending.extend(graph.get(a,set())-seen)
        return any(a in seen and t==target for a,t in local.items())

    def predicate_match(self,sub,p):
        w=sub.args.get('where')
        if not w:return False
        expected=p.get('value')
        for n in _parts(w.this):
            if 'sql' in p:
                template=parse('SELECT 1 WHERE '+p['sql']).args['where'].this
                if type(n) is not type(template):continue
                if [text(v) for v in n.find_all(exp.Literal)]!=[text(v) for v in template.find_all(exp.Literal)]:continue
            else:
                if type(n).__name__!=p['operator'] or not isinstance(n.expression,exp.Literal) or n.expression.this!=str(expected):continue
            cs=list(n.find_all(exp.Column))
            if len(cs)==1 and owner(cs[0],sub,self.schema)==(p['table'],p['column']):return True
        return False

    def count_support(self,count,select,target):
        arg=count.this;distinct=isinstance(arg,exp.Distinct)
        if distinct:arg=arg.expressions[0] if len(arg.expressions)==1 else None
        if isinstance(arg,exp.Star):
            ss=sources(select)
            return bool(ss and list(ss.values())==[target] and not select.args.get('joins'))
        return isinstance(arg,exp.Column) and owner(arg,select,self.schema)==(target,pk(self.schema,target)) and (distinct or not select.args.get('joins'))

    def goal_state(self,tree,g):
        if not isinstance(tree,exp.Select):
            if g['family']=='R2' and isinstance(tree,exp.Intersect):
                # Only recognize two independently bound branches at the same root grain.
                pair=g['branches'][0] if len(g['branches'])==1 else None
                if pair:
                    for root in g['roots']:
                        if all(self.root_match(s,root) and self.predicate_match(s,p) and
                               (p['table']==root['table'] or self.flat_linked(s,root,p['table'])) for s,p in zip((tree.this,tree.expression),pair)):
                            return 'satisfied' if all(p.get('known',True) for p in pair) else 'unknown'
            return 'unknown'
        if g['family']=='R5':
            aggs=[n for e in tree.expressions for n in e.find_all(exp.AggFunc)]
            ss=sources(tree)
            if not ss or len(aggs)!=1:return 'unknown'
            arg=aggs[0].this;arg=arg.this if isinstance(arg,exp.Cast) else arg
            if type(aggs[0]).__name__!=g['operation'] or not isinstance(arg,exp.Column) or owner(arg,tree,self.schema)!=(g['table'],g['column']):return 'unknown'
            root={'kind':'entity','table':g['table'],'key':pk(self.schema,g['table']),'certain':True}
            exists=[p for p in _parts(tree.args['where'].this) if isinstance(p,exp.Exists)] if tree.args.get('where') else []
            linked=any(self.linked(e.this,tree,root,t) for e in exists for t in (sources(e.this) or {}).values())
            return 'satisfied' if list(ss.values())==[g['table']] and linked else 'unknown'
        roots=[r for r in g['roots'] if self.root_match(tree,r)]
        if not roots:return 'unknown'
        where=tree.args.get('where');parts=_parts(where.this if where else None)
        for root in roots:
            if g['family']=='R1':
                for n in parts:
                    if not isinstance(n,exp.Not) or not isinstance(n.this,exp.Exists):continue
                    sub=n.this.this
                    for target in g['relations']:
                        ps=[p for p in g['predicates'] if p['table']==target]
                        if self.linked(sub,tree,root,target) and (not ps or any(self.predicate_match(sub,p) for p in ps)):
                            return 'satisfied'
                # A definite anti-goal represented only as a positive existence is opposite.
                if any(isinstance(p,exp.Exists) for p in parts):return 'contradicted'
            elif g['family']=='R2':
                exists=[p.this for p in parts if isinstance(p,exp.Exists)]
                for pair in g['branches']:
                    matched=[]
                    for p in pair:matched.append([i for i,s in enumerate(exists) if self.linked(s,tree,root,p['table']) and self.predicate_match(s,p)])
                    if len(matched)==2 and any(a!=b for a in matched[0] for b in matched[1]):return 'satisfied' if all(p.get('known',True) for p in pair) else 'unknown'
                    # Same local field cannot equal two different values on the same row.
                    if all(p.get('known',True) for p in pair) and pair[0]['table']==pair[1]['table'] and pair[0]['column']==pair[1]['column']:
                        if all(self.predicate_match(tree,p) for p in pair):return 'contradicted'
                if any(isinstance(p,exp.Or) for p in parts):return 'contradicted'
            elif g['family']=='R3':
                target=g['count_table'];quantity=g['quantity']
                if root['kind']=='domain':
                    group=tree.args.get('group')
                    if not group or not any(isinstance(c,exp.Column) and owner(c,tree,self.schema)==(root['table'],root['key']) for c in group.expressions):continue
                    if not self.flat_linked(tree,root,target) and root['table']!=target:continue
                    if quantity:
                        conditions=self.count_conditions(tree,target)
                        relevant=[(n,c,s) for n,c,s in conditions if isinstance(n.expression,exp.Literal) and n.expression.this==str(quantity['number'])]
                        if relevant:return 'satisfied' if any(type(n).__name__==quantity['operator'] for n,c,s in relevant) else 'contradicted'
                    elif any(self.count_support(c,tree,target) for e in tree.expressions for c in e.find_all(exp.Count)):return 'satisfied'
                else:
                    for sub in tree.find_all(exp.Subquery):
                        s=sub.this
                        if isinstance(s,exp.Select) and self.linked(s,tree,root,target) and any(self.count_support(c,s,target) for c in s.find_all(exp.Count)):return 'satisfied'
            elif g['family']=='R4':
                found=[];bad=False
                for cgoal in g['counts']:
                    hits=[]
                    for n,c,s in self.count_conditions(tree,cgoal['table']):
                        if not isinstance(n.expression,exp.Literal) or n.expression.this!=str(cgoal['number']):continue
                        if s is not tree and not self.linked(s,tree,root,cgoal['table']):continue
                        if type(n).__name__!=cgoal['operator']:bad=True;continue
                        hits.append(n)
                    found.append(hits)
                if all(found):
                    # Match the common Boolean node of the two independent count conditions.
                    for a in found[0]:
                        for b in found[1]:
                            if a is b:continue
                            ancestors_a=[];p=a.parent
                            while p is not None and p is not tree:ancestors_a.append(p);p=p.parent
                            p=b.parent
                            while p is not None and p is not tree:
                                if any(p is x for x in ancestors_a):
                                    if type(p).__name__.upper()==g['join']:return 'satisfied'
                                    if isinstance(p,(exp.And,exp.Or)):return 'contradicted'
                                    break
                                p=p.parent
                if bad:return 'contradicted'
        return 'unknown'

    def flat_linked(self,tree,root,target):
        ss=sources(tree)
        if not ss:return False
        graph={};conditions=[j.args.get('on') for j in tree.args.get('joins',[])]+[tree.args['where'].this if tree.args.get('where') else None]
        for condition in conditions:
            for n in _parts(condition):
                if not isinstance(n,exp.EQ) or not isinstance(n.this,exp.Column) or not isinstance(n.expression,exp.Column):continue
                a=owner(n.this,tree,self.schema);b=owner(n.expression,tree,self.schema)
                if a and b and (*a,*b) in self.schema.edges.get(a[0],[]):graph.setdefault(a[0],set()).add(b[0]);graph.setdefault(b[0],set()).add(a[0])
        seen=set();pending=[root['table']]
        while pending:
            t=pending.pop()
            if t in seen:continue
            seen.add(t);pending.extend(graph.get(t,set())-seen)
        return target in seen

    def count_conditions(self,tree,target):
        found=[]
        for n in tree.walk():
            if not isinstance(n,(exp.EQ,exp.GT,exp.GTE,exp.LT,exp.LTE)):continue
            arg=n.this;s=n.find_ancestor(exp.Select)
            if isinstance(arg,exp.Subquery) and isinstance(arg.this,exp.Select) and len(arg.this.expressions)==1:s=arg.this;arg=unalias(s.expressions[0])
            if isinstance(arg,exp.Count) and self.count_support(arg,s,target):found.append((n,arg,s))
        return found
