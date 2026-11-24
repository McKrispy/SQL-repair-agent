"""task obligations and explicit soft relation risks; no grade-derived features."""
import re
from sqlglot import exp
from mp3_agent.semantic_facts import SemanticFacts
from mp3_agent.rule_support import header_expression, output_states, output_branches, unalias, text
from mp3_agent.execution_identity import bound_query, physical_expression


class AuditFacts(SemanticFacts):
    def __init__(self, question, schema, example, config):
        super().__init__(question, schema, example)
        self.config = config
        self.aggregate_intent = bool(re.search(r'\b(?:' + '|'.join(re.escape(x) for x in config['aggregate_phrases']) + r')\b', self.q))
        self.aggregate_intent |= any(isinstance(header_expression(x), exp.AggFunc) for x in example['columns'])
        self.count_distinct = bool(self.count_phrase and re.match(r'(?:distinct|different|unique)\b', self.count_phrase))
        if self.count_table and self.count_distinct:
            self.obligations.append({'id':'count_distinct_identity','table':self.count_table,'source':self.count_phrase})

    def count_identity(self, tree):
        result = {'state':'unknown','target':self.count_table,'identity':None,'distinct':None,'reason':'unresolved_count'}
        if not self.count_table or not isinstance(tree, exp.Select): return result
        counts = [n for n in tree.find_all(exp.Count) if n.find_ancestor(exp.Select) is tree]
        if len(counts) != 1: return result
        n = counts[0].this; result['distinct'] = isinstance(n, exp.Distinct)
        if isinstance(n, exp.Distinct): n = n.expressions[0] if len(n.expressions) == 1 else None
        if isinstance(n, exp.Star):
            source = tree.args.get('from')
            if source and isinstance(source.this, exp.Table) and source.this.name.casefold() == self.count_table and not tree.args.get('joins'):
                result.update(state='satisfied', identity=self.count_table, reason='target_single_table_rows')
            return result
        if not isinstance(n, exp.Column): return result
        owner = self.owner(n, tree); name = n.name.casefold()
        target_keys = [k for k,c in self.schema.columns[self.count_table].items() if c.get('pk')]
        if owner and len(target_keys) == 1:
            direct_pk = (owner, name) == (self.count_table, target_keys[0])
            direct_fk = any(e == (owner, name, self.count_table, target_keys[0]) for e in self.schema.edges.get(owner, []))
            # Schema.edges is bidirectional. Only an actually declared outgoing FK is evidence.
            direct_fk = direct_fk and any(f['from'].casefold()==name and f['table'].casefold()==self.count_table and f['to'].casefold()==target_keys[0]
                                          for f in self.schema.tables[owner].get('foreign_keys', []))
            if direct_pk or direct_fk:
                result.update(state='satisfied',identity=self.count_table,reason='target_primary_key' if direct_pk else 'direct_declared_foreign_key',
                              physical=[owner,name], nullable_or_multiplicity_unproved=True)
            elif self.schema.columns[owner][name].get('pk'):
                result['reason'] = 'other_entity_identifier'
        return result

    def evaluate(self, tree):
        result = super().evaluate(tree); count = self.count_identity(tree)
        for e in result:
            if e['id']=='count_entity': e.update(state=count['state'], detail=count)
            if e['id']=='count_distinct_identity':
                e['state'] = ('satisfied' if count['distinct'] else 'contradicted') if count['state']=='satisfied' else 'unknown'
            if e['id']=='grouping' and not self.aggregate_intent: e['state']='unknown'
        return result

    def context(self, tree, rid, focus):
        guard, roles = super().context(tree, rid, focus)
        if rid in {25,122,172,181} and not self.aggregate_intent:
            guard['intent'] = {'state':None,'evidence':{'reason':'no_positive_task_aggregate_signal'}}
        return guard, roles

    def output_vector(self, tree):
        headers = [header_expression(x) for x in self.example['columns']]
        arity, _ = output_states(tree, [None]*len(headers))
        branches = output_branches(tree)
        states = ['unknown']*len(headers)
        for i,h in enumerate(headers):
            expected = self.header_signature(h)
            if expected is None or not branches or any(any(e.is_star for e in b) for b in branches):continue
            if any(i>=len(b) for b in branches):states[i]='contradicted';continue
            nodes = [unalias(b[i]) for b in branches]
            if isinstance(h,exp.Count):
                # Only function/position, never COUNT argument/DISTINCT identity.
                states[i]='satisfied' if all(isinstance(n,exp.Count) for n in nodes) else 'contradicted' if all(isinstance(n,exp.AggFunc) for n in nodes) else 'unknown'
                continue
            actual = [self.output_signature(n) for n in nodes]
            if any(x is None for x in actual) or len(set(actual))!=1:continue
            states[i]='satisfied' if actual[0]==expected else 'contradicted'
        return [{'id':'output_count','state':arity}] + [{'id':f'output_{i}','state':s} for i,s in enumerate(states)]

    def header_signature(self, header):
        """Canonical physical meaning only for schema-resolvable SQL-like labels."""
        if header is None:return None
        node=header.copy()
        for c in node.find_all(exp.Column):
            if c.is_star or not c.name.isascii() or (c.table and not c.table.isascii()) or c.db or c.catalog:return None
            owners=[t for t,cs in self.schema.columns.items() if c.name.casefold() in cs and (not c.table or t==c.table.casefold())]
            if len(owners)!=1 or not owners[0].isascii():return None
            c.set('this',exp.to_identifier(c.name.casefold(),quoted=True))
            c.set('table',exp.to_identifier(owners[0],quoted=True))
        return text(node)

    def output_signature(self, node):
        if not isinstance(node,(exp.Column,exp.AggFunc)):return None
        select=node.find_ancestor(exp.Select)
        if select is None:return None
        try:
            q,bindings,reason=bound_query(text(select),self.schema)
            if reason or any(not t.name.isascii() for t in q.find_all(exp.Table)) or any(not c.name.isascii() for c in q.find_all(exp.Column)):
                return None
            return physical_expression(node,select,self.schema)
        except (ValueError,RecursionError):return None

    def relation_risks(self, tree):
        try:
            q, bindings, reason = bound_query(text(tree),self.schema)
            if reason: return []
            risks=[]
            for join in q.args.get('joins',[]):
                if join.args.get('side') or join.args.get('kind') not in (None,'INNER'): continue
                on=join.args.get('on')
                if not isinstance(on,exp.EQ) or not isinstance(on.this,exp.Column) or not isinstance(on.expression,exp.Column): continue
                a,c=bindings[id(on.this)]; b,d=bindings[id(on.expression)]
                if a==b or (a,c,b,d) in self.schema.edges.get(a,[]): continue
                def role(t,n):
                    return self.schema.columns[t][n].get('pk') or any(e[1]==n for e in self.schema.edges.get(t,[]))
                if not role(a,c) or not role(b,d): continue
                # Bounded connectivity proof, not a shortest-path preference.
                frontier=[(a,{a},[])] ; found=None
                while frontier:
                    t,seen,path=frontier.pop()
                    if t==b: found=path; break
                    if len(path)>=self.config['skeleton_max_edges']: continue
                    for edge in sorted(self.schema.edges.get(t,[]),reverse=True):
                        if edge[2] not in seen: frontier.append((edge[2],seen|{edge[2]},path+[edge]))
                if found:
                    risks.append({'id':'join_key_disagreement','pair':[a,c,b,d],'endpoints':sorted([a,b]),'alternative':found,'business_relation':'unknown'})
            return risks
        except (ValueError,RecursionError): return []

    def preserves_groups(self, before, after):
        if not isinstance(before,exp.Select) or not isinstance(after,exp.Select): return None
        old=before.args.get('group'); new=after.args.get('group')
        if old is None: return True
        a=[physical_expression(x,before,self.schema) for x in old.expressions]
        b=[physical_expression(x,after,self.schema) for x in new.expressions] if new else []
        if None in a or None in b: return None
        return set(a)<=set(b)


def weak_feedback(feedback):
    if not feedback or not feedback.get('ok') or not feedback.get('matches_example') or feedback.get('truncated'): return False
    rows=feedback.get('rows')
    return feedback.get('row_count')==0 or (feedback.get('row_count')==1 and isinstance(rows,list) and len(rows)==1 and bool(rows[0]) and all(x is None for x in rows[0]))
