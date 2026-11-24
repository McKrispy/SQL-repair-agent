"""2026-11-21 12:53 HKT. Deterministic, causal facts; no labels or task IDs."""
import re
from sqlglot import exp
from sqlglot.errors import SqlglotError
from mp3_agent.rule_support import Schema, scopes, unalias
from mp3_agent.semantic_ops import normalize, parse, text, relation, same, columns_from_equality


def words(s):
    def singular(w):
        if w.endswith('ies'):return w[:-3]+'y'
        if w.endswith('s') and not w.endswith(('ss','us')):return w[:-1]
        return w
    return tuple(singular(w) for w in re.findall(r'[a-z0-9]+',s.casefold().replace('_',' ')))


class SemanticFacts:
    def __init__(self,question,schema,example):
        self.question=question;self.q=question.casefold();self.schema=schema;self.example=example
        self.occurrences=bool(re.search(r'\b(rows|records|entries|occurrences)\b',self.q))
        self.nonnull=bool(re.search(r'\bnon[ -]null\b',self.q))
        self.preserve_null=bool(re.search(r'\b(?:including|include|even)\b[^.]*\b(?:null|missing)\b',self.q))
        self.distinct=bool(re.search(r'\b(distinct|different|unique)\b',self.q))
        masked=re.sub(r'\bno more than\b','at most',self.q)
        self.negative=bool(re.search(r'\bnever\b|\bwithout\b|\bnot\b.+\bany\b|\b(?:do|does) not have\b|\bno\b',masked))
        self.both=bool(re.search(r'\bboth\b.+\band\b',self.q))
        count=re.search(r'\b(?:how many|number of|count of)\s+(?:the\s+)?((?:(?:distinct|different|unique)\s+)?[a-z_ ]+)',self.q)
        self.count_phrase=count.group(1) if count else None
        self.count_table=self.match_table(self.count_phrase,prefix=True) if self.count_phrase else None
        self.root_distinct=bool(re.search(r'^(?:what are|list|show|return|give(?: me)?)\s+(?:the\s+)?(?:distinct|different|unique)\b',self.q))
        group=re.search(r'\b(?:for each|per|grouped by)\s+(?:the\s+)?([a-z_ ]+)',self.q)
        self.group_phrase=group.group(1) if group else None
        self.group_table=self.match_table(self.group_phrase,prefix=True) if self.group_phrase else None
        self.group_field=self.match_field(self.group_phrase) if self.group_phrase else None
        self.obligations=[]
        if self.count_table:self.obligations.append({'id':'count_entity','table':self.count_table,'source':self.count_phrase})
        if self.root_distinct:self.obligations.append({'id':'output_distinct','source':'explicit output distinct phrase'})
        if self.occurrences:self.obligations.append({'id':'occurrences','source':'explicit rows/records/entries'})
        if self.group_phrase and (self.group_table or self.group_field):self.obligations.append({'id':'grouping','source':self.group_phrase})
        if self.negative:self.obligations.append({'id':'absence','source':'explicit entity-negative phrase'})
        if self.both:self.obligations.append({'id':'both','source':'both X and Y'})

    def match_table(self,phrase,prefix=False):
        if not phrase:return None
        w=words(phrase)
        if w and w[0] in ('distinct','different','unique'):w=w[1:]
        hits=[t for t in self.schema.tables if (w[:len(words(t))]==words(t) if prefix else w==words(t))]
        return hits[0] if len(hits)==1 else None

    def match_field(self,phrase):
        w=words(phrase or '')
        hits=[(t,c) for t,columns in self.schema.columns.items() for c in columns if w[:len(words(c))]==words(c)]
        return hits[0] if len(hits)==1 else None

    def owner(self,col,select):
        frames=[s for s in scopes(select) if s.expression is select]
        if len(frames)!=1:return None
        sources=self.schema.sources(frames[0]);name=col.name.casefold()
        if col.table:
            pair=sources.get(col.table.casefold())
            return pair[0] if pair and name in pair[1] else None
        hits=[p[0] for p in sources.values() if name in p[1]]
        return hits[0] if len(hits)==1 else None

    def schema_state(self,tree):
        unknown=False;bad=[]
        for scope in scopes(tree):
            if not isinstance(scope.expression,exp.Select):continue
            sources=self.schema.sources(scope)
            if not sources and scope.references:unknown=True
            aliases={x.alias.casefold() for x in scope.expression.expressions if x.alias}
            try:columns=scope.columns
            except SqlglotError:
                # Lazy scope properties can fail even after traverse_scope succeeds.
                # Ambiguous/duplicate aliases never prove a guard or abort a bank.
                unknown=True
                continue
            for col in columns:
                if col.is_star:continue
                if col.table:
                    src=sources.get(col.table.casefold())
                    if src and col.name.casefold() not in src[1]:bad.append(text(col))
                    elif not src:
                        parent=scope.parent;found=False
                        while parent:
                            outer=self.schema.sources(parent)
                            if col.table.casefold() in outer:
                                found=col.name.casefold() in outer[col.table.casefold()][1];break
                            parent=parent.parent
                        if not found:unknown=True
                else:
                    hits=sum(col.name.casefold() in p[1] for p in sources.values())
                    if hits!=1 and col.name.casefold() not in aliases:
                        unknown=True
                        if hits==0 and not scope.parent:bad.append(text(col))
        return (False if bad else None if unknown else True),bad

    def group_column(self,select):
        source=relation(select)
        if not isinstance(source,exp.Table):return None
        table=source.name.casefold();alias=source.alias_or_name
        if self.group_field and self.group_field[0]==table:return exp.column(self.group_field[1],table=alias)
        if self.group_table==table:
            keys=[c['name'] for c in self.schema.columns[table].values() if c.get('pk')]
            if len(keys)==1:return exp.column(keys[0],table=alias)
        return None

    def evaluate(self,tree):
        result=[];root=tree if isinstance(tree,exp.Select) else None
        for o in self.obligations:
            state='unknown';k=o['id']
            if k=='count_entity' and root:
                counts=[n for n in root.find_all(exp.Count) if n.find_ancestor(exp.Select) is root]
                if len(counts)==1:
                    n=counts[0].this
                    if isinstance(n,exp.Distinct) and len(n.expressions)==1:n=n.expressions[0]
                    if isinstance(n,exp.Column):
                        owner=self.owner(n,root)
                        if owner:state='satisfied' if owner==self.count_table else 'contradicted'
                    elif isinstance(n,exp.Star):
                        src=relation(root)
                        if isinstance(src,exp.Table):
                            state=('satisfied' if not root.args.get('joins') else 'unknown') if src.name.casefold()==self.count_table else 'contradicted'
            elif k=='output_distinct' and root:state='satisfied' if root.args.get('distinct') else 'contradicted'
            elif k=='occurrences' and root:state='contradicted' if root.args.get('distinct') else 'satisfied'
            elif k=='grouping' and root:
                col=self.group_column(root);g=root.args.get('group')
                if col is not None:state='satisfied' if g and any(same(x,col) for x in g.expressions) else 'contradicted'
            elif k=='absence' and root:
                found=[]
                for n in root.find_all(exp.Not):
                    if not isinstance(n.this,exp.Exists) or not isinstance(n.this.this,exp.Select):continue
                    sub=n.this.this;w=sub.args.get('where');outer=relation(root)
                    if w and isinstance(outer,exp.Table) and self.literal_intent(w) and any(isinstance(eq,exp.EQ) and any(c.table.casefold()==outer.alias_or_name.casefold() for c in eq.find_all(exp.Column)) for eq in w.find_all(exp.EQ)):found.append(n)
                if found:state='satisfied'
                elif root.args.get('where') and (list(root.args['where'].find_all(exp.NEQ)) or list(root.args['where'].find_all(exp.Exists))):state='contradicted'
            elif k=='both':
                if isinstance(tree,exp.Intersect):state='satisfied'
                elif isinstance(tree,exp.Union):state='contradicted'
                elif root and root.args.get('where'):
                    w=root.args['where'];exists=list(w.find_all(exp.Exists))
                    # Two structurally distinct correlated existence conditions, not a lineage flag.
                    outer=relation(root)
                    correct=bool(isinstance(outer,exp.Table) and len(exists)==2)
                    for e in exists:
                        sub=e.this;condition=sub.args.get('where') if isinstance(sub,exp.Select) else None
                        correct=correct and bool(condition and self.literal_intent(condition) and any(c.table.casefold()==outer.alias_or_name.casefold() for c in condition.find_all(exp.Column)))
                    if correct and text(exists[0])!=text(exists[1]) and not list(w.find_all(exp.Or)):state='satisfied'
                    elif list(w.find_all(exp.Or)):state='contradicted'
            result.append({'id':k,'state':state,'evidence':o})
        return result

    def literal_intent(self,node):
        number_words={'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
        for n in node.find_all(exp.Literal):
            select=n.find_ancestor(exp.Select)
            if not n.is_string and n.this in {'0','1'} and select and any(unalias(e) is n for e in select.expressions):continue
            if str(n.this).casefold() not in self.q and not (not n.is_string and any(v==str(n.this) and k in self.q.split() for k,v in number_words.items())):return False
        return True

    def context(self,tree,rid,focus):
        """Return independently recorded guard states and roles for this operation."""
        structure=isinstance(focus,exp.Select) or (rid==30 and isinstance(focus,exp.Union))
        schema_state,bad=self.schema_state(tree)
        roles={'question':self.question};intent=None;data=True;reason='intent not established'
        root=focus if isinstance(focus,exp.Select) else None
        source=relation(root) if root else None
        if rid==30:
            intent=self.both;reason='explicit both, set or same-attribute alternatives'
        elif root:
            gc=self.group_column(root)
            if gc is not None:roles['group_column']=text(gc)
            if rid in {21,52}:
                intent=self.root_distinct if rid==21 else self.occurrences;reason='explicit distinct' if rid==21 else 'explicit bag occurrences'
            elif rid in {31,180,39}:
                intent=self.occurrences;reason='explicit row occurrences; no DISTINCT inference from absence'
            elif rid==183:
                intent=bool(self.distinct and self.count_table)
                counts=[n for n in root.find_all(exp.Count) if n.find_ancestor(exp.Select) is root and isinstance(n.this,exp.Column)]
                if len(counts)!=1 or self.owner(counts[0].this,root)!=self.count_table:intent=None
                reason='explicit distinct counted entity'
            elif rid==86:
                if self.nonnull:
                    cols=[c for c in tree.find_all(exp.Column) if c.find_ancestor(exp.Select) is root]
                    if len({text(c) for c in cols})==1 and cols:roles['count_column']=text(cols[0]);intent=True
                reason='explicit non-null count with unique identified measure'
            elif rid in {25,122,172,181}:
                intent=gc is not None and bool(self.group_phrase)
                if rid==25:intent=bool(intent and re.search(r'\bgroups?\b',self.q))
                reason='explicit per/each/grouped-by identity'
            elif rid in {72,157}:
                intent=bool(re.search(r'\b(?:whose|having|with)\s+(?:the\s+)?(?:sum|total|average|avg)\b',self.q)) if rid==72 else bool(re.search(r'\b(total|sum|average|avg|count|number)\b',self.q))
                reason='explicit aggregate filter and existing same-scope expression'
            elif rid in {59,100,145,173}:
                intent=self.negative;reason='explicit entity absence'
                if rid in {145,173}:
                    # Outer NULL semantics are a real obligation, not supplied by a witness.
                    data=self.preserve_null
                    if not data and isinstance(source,exp.Table):
                        cols=list(root.find_all(exp.In));left=cols[0].this if len(cols)==1 else None
                        owner=self.owner(left,root) if isinstance(left,exp.Column) else None
                        data=True if owner and self.schema.columns[owner][left.name.casefold()].get('not_null') else None
            elif rid in {106,132,188}:
                intent=bool(self.negative or re.search(r'\b(has|have|with|having|exists?)\b',self.q))
                reason='explicit relation existence'
                es=[x for x in root.find_all(exp.Exists) if x.find_ancestor(exp.Select) is root]
                if rid in {106,132} and len(es)==1 and isinstance(es[0].this,exp.Select):
                    inner=relation(es[0].this)
                    if isinstance(source,exp.Table) and isinstance(inner,exp.Table):
                        path=self.schema.unique_path(source.name.casefold(),inner.name.casefold())
                        if path and len(path)==1:
                            roles.update(outer_column=text(exp.column(path[0][1],table=source.alias_or_name)),inner_column=text(exp.column(path[0][3],table=inner.alias_or_name)))
                        elif path and len(path)==2 and rid==132:
                            alias='_s1_bridge'
                            used={t.alias_or_name.casefold() for t in tree.find_all(exp.Table)}
                            while alias.casefold() in used:alias+='_' 
                            bridge=exp.Table(this=exp.to_identifier(self.schema.tables[path[0][2]]['name']),alias=exp.TableAlias(this=exp.to_identifier(alias)))
                            on=exp.EQ(this=exp.column(path[1][1],table=alias),expression=exp.column(path[1][3],table=inner.alias_or_name))
                            query=exp.select('1').from_(inner.copy());query.set('joins',[exp.Join(this=bridge,on=on)])
                            roles.update(outer_column=text(exp.column(path[0][1],table=source.alias_or_name)),inner_column=text(exp.column(path[0][3],table=alias)),bridge_join_sql=text(query))
                        else:data=None
            elif rid in {23,190}:
                intent=bool(re.search(r'\bat least one\b|\b(?:has|have) (?:a|an)\b|\bwith\b',self.q))
                if rid==190:intent=bool(intent and self.count_table and isinstance(source,exp.Table) and source.name.casefold()==self.count_table)
                else:intent=bool(intent and re.search(r'\b(sum|total)\b',self.q))
                reason='left occurrences/measure with a right-existence condition'
            elif rid==113:
                joins=root.args.get('joins',[])
                if len(joins)==1 and joins[0].args.get('side')=='LEFT':
                    pair=columns_from_equality(joins[0])
                    if pair:
                        roles['count_column']=text(pair[1]);intent=joins[0].this.name.casefold()==self.count_table
                        reason='count right entity; join equality proves matched key non-null'
            if rid==100:
                joins=root.args.get('joins',[])
                data=bool(len(joins)==1 and columns_from_equality(joins[0]))
        if root and rid in {59,72,157}:
            conditions=[x for x in (root.args.get('where'),root.args.get('having')) if x]
            if any(not self.literal_intent(x) for x in conditions):intent=None
        explicit_opposite=(rid==21 and self.occurrences and not self.root_distinct) or (rid==52 and self.root_distinct)
        intent_state=True if intent else False if explicit_opposite else None
        return {'structure':{'state':structure,'evidence':{'node':type(focus).__name__}},
                'schema':{'state':schema_state,'evidence':{'source':'charged schema','bad_columns':bad}},
                'intent':{'state':intent_state,'evidence':{'source':'observation.task.question','reason':reason}},
                'data':{'state':data,'evidence':{'source':'AST/schema facts','preserve_null':self.preserve_null}}},roles


def improves(old,new):
    """Component-wise protection: unknown is never a completed contradiction repair."""
    a={x['id']:x['state'] for x in old};b={x['id']:x['state'] for x in new}
    if set(a)!=set(b):return False
    gain=False
    for k,v in a.items():
        w=b[k]
        if v=='satisfied' and w!='satisfied':return False
        if v=='unknown' and w=='contradicted':return False
        if v=='contradicted' and w=='unknown':return False
        gain|=w=='satisfied' and v!='satisfied'
    return gain
