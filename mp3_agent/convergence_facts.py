"""typed task evidence; only current observation and paid schema."""
from sqlglot import exp
from sqlglot.errors import SqlglotError
from mp3_agent.audit_facts import AuditFacts
from mp3_agent.execution_identity import bound_query,physical_expression
from mp3_agent.rule_support import parse,text,quantity,count_comparisons
from mp3_agent.skeleton_rebuild import _prepare,_parts,_sources,_physical_owner
from mp3_agent.convergence_rescue import grain_spec,name_spec


class ConvergenceFacts(AuditFacts):
    def __init__(self,question,schema,example,config,original=''):
        super().__init__(question,schema,example,config)
        self.quantity=quantity(question)
        self.grain=grain_spec(question,example,schema,config)
        self.name=name_spec(question,original,schema,config)
        if self.quantity:self.obligations.append({'id':'count_boundary','source':self.quantity['evidence']})
        if self.grain:self.obligations.append({'id':'entity_group_count','source':self.grain})
        if self.name:self.obligations.append({'id':'entity_name_binding','source':self.name})

    def prepared(self,tree):
        try:
            return _prepare(tree,self.schema) if isinstance(tree,exp.Select) and tree.args.get('from') else tree
        except (ValueError,RecursionError,SqlglotError):return None

    def output_vector(self,tree):
        prepared=self.prepared(tree)
        return super().output_vector(prepared if prepared is not None else tree)

    def evaluate(self,tree):
        result=super().evaluate(tree)
        prepared=self.prepared(tree)
        root=prepared if isinstance(prepared,exp.Select) else None
        for e in result:
            if e['id']=='grouping' and self.aggregate_intent and root:
                target=self.group_column(root)
                if target is not None:
                    owner=self.owner(target,root);g=root.args.get('group')
                    if owner:
                        expected=text(exp.column(target.name.casefold(),table=owner,quoted=True))
                        actual=[physical_expression(x,root,self.schema) for x in g.expressions] if g else []
                        e['state']='satisfied' if expected in actual else 'unknown' if None in actual else 'contradicted'
        if self.quantity:
            found=count_comparisons(tree,self.quantity)
            state=('satisfied' if all(type(n) is self.quantity['operator'] for n in found) else 'contradicted') if found else 'unknown'
            result.append({'id':'count_boundary','state':state,'evidence':self.quantity['evidence']})
        if self.grain:result.append({'id':'entity_group_count','state':self.grain_state(root),'evidence':self.grain})
        if self.name:result.append({'id':'entity_name_binding','state':self.name_state(root),'evidence':self.name})
        return result

    def grain_state(self,root):
        if root is None:return 'unknown'
        s=self.grain;f=root.args.get('from');g=root.args.get('group');having=root.args.get('having')
        if not f or not isinstance(f.this,exp.Table) or f.this.name.casefold()!=s['table'] or root.args.get('joins') or not g or not having:return 'unknown'
        if not any(isinstance(c,exp.Column) and self.owner(c,root)==s['table'] and c.name.casefold()==s['group'] for c in g.expressions):return 'unknown'
        nodes=count_comparisons(root,{'noun':s['table'],'number':s['number'],'operator':getattr(exp,s['operator'])})
        # count_comparisons uses the question noun; table words can contain separators.
        if not nodes:nodes=[n for n in having.walk() if n.find_ancestor(exp.Select) is root and isinstance(n,(exp.EQ,exp.GT,exp.GTE,exp.LT,exp.LTE)) and
            isinstance(n.this,exp.Count) and isinstance(n.expression,exp.Literal) and not n.expression.is_string and n.expression.this==str(s['number'])]
        if len(nodes)!=1:return 'unknown'
        count=nodes[0].this;arg=count.this
        if isinstance(arg,exp.Distinct):arg=arg.expressions[0] if len(arg.expressions)==1 else None
        identity=isinstance(arg,exp.Star) or isinstance(arg,exp.Column) and self.owner(arg,root)==s['table'] and arg.name.casefold()==s['pk']
        if not identity:return 'unknown'
        return 'satisfied' if type(nodes[0]).__name__==s['operator'] else 'contradicted'

    def name_state(self,root):
        if root is None:return 'unknown'
        w=root.args.get('where')
        if not w:return 'unknown'
        for part in _parts(w.this):
            if isinstance(part,exp.EQ) and isinstance(part.this,exp.Column) and isinstance(part.expression,exp.Literal) and part.expression.is_string:
                if self.owner(part.this,root)==self.name['table'] and part.this.name.casefold()==self.name['column'] and part.expression.this==self.name['literal']:
                    return 'satisfied'
        return 'unknown'

    def unconnected(self,tree):
        try:
            q,bindings,reason=bound_query(text(tree),self.schema)
            if reason or any(j.args.get('side') or j.args.get('kind') not in (None,'INNER') for j in q.args.get('joins',[])):return []
            conditions=[j.args.get('on') for j in q.args.get('joins',[])]
            if q.args.get('where'):conditions.append(q.args['where'].this)
            if any(n is None or list(n.find_all(exp.Or)) for n in conditions):return []
            graph={};sources=_sources(q)
            for condition in conditions:
                for part in _parts(condition):
                    owners={bindings[id(c)][0] for c in part.find_all(exp.Column)}
                    for a in owners:
                        graph.setdefault(a,set()).update(owners-{a})
            root=q.args['from'].this.name.casefold();seen={root};pending=[root]
            while pending:
                n=pending.pop()
                for other in graph.get(n,set())-seen:seen.add(other);pending.append(other)
            return [{'id':'unconnected_relation','table':t,'evidence':'no_known_cross_source_condition'} for t in sorted(set(sources.values())-seen)]
        except (ValueError,RecursionError,SqlglotError):return []

    def missing_count_scope(self,tree):
        if not self.count_table:return 0
        if self.count_identity(tree)['state']=='satisfied':return 0
        return int(not any(t.name.casefold()==self.count_table for t in tree.find_all(exp.Table)))
