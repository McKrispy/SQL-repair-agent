"""2026-11-21 12:53 HKT. SQLGlot edits for the approved corpus subset.

This module performs no IO, matching of questions, database execution or scoring.
Roles and intent are established separately; missing roles never use column order.
"""
from sqlglot import exp
from mp3_agent.rule_support import parse, text, unalias, local_columns
from mp3_agent.rule_rewrites import replace_at

RULE_IDS = (21, 23, 25, 30, 31, 39, 52, 59, 72, 86, 100, 106, 113,
            122, 132, 145, 157, 172, 173, 180, 181, 183, 188, 190)
FAMILIES = {i: ('negative' if i in {59,100,106,132,145,173,188} else
                'intersection' if i == 30 else 'multiplicity' if i in {21,31,52,86,113,180,183}
                else 'fanout' if i in {23,190} else 'grouping') for i in RULE_IDS}


def same(a, b):
    def key(n):
        n=unalias(n).copy()
        for i in n.find_all(exp.Identifier):
            i.set('quoted',False);i.set('this',i.this.casefold())
        return text(n)
    return a is not None and b is not None and key(a)==key(b)


def relation(select):
    f = select.args.get('from')
    return f.this if f is not None else None


def selects(tree):
    return [n for n in tree.find_all(exp.Select)]


def direct(node, kind, scope):
    return [n for n in node.find_all(kind) if n.find_ancestor(exp.Select) is scope]


def role_column(roles, name):
    value = roles.get(name)
    return parse('SELECT ' + value).expressions[0] if value else None


def normalize(tree):
    """Remove AST grouping wrappers without changing boolean or JOIN association."""
    tree=tree.copy()
    for n in reversed(list(tree.find_all(exp.Paren))):
        n.replace(n.this.copy())
    for q in tree.find_all(exp.Select):
        source=relation(q)
        if isinstance(source,exp.Subquery) and isinstance(source.this,exp.Table) and not q.args.get('joins'):
            table=source.this.copy();joins=table.args.get('joins',[])
            if len(joins)==1:
                table.set('joins',None);q.set('from',exp.From(this=table));q.set('joins',[j.copy() for j in joins])
    return tree


def columns_from_equality(join):
    on = join.args.get('on')
    if not isinstance(on, exp.EQ) or not all(isinstance(x, exp.Column) for x in (on.this, on.expression)):
        return None
    right = join.this.alias_or_name.casefold()
    a, b = on.this, on.expression
    return (b, a) if a.table.casefold() == right else (a, b) if b.table.casefold() == right else None


def edit(tree, rule_id, roles, focus=None):
    """Return AST variants. Structural nonmatches and unbound roles return none."""
    if rule_id not in RULE_IDS:
        raise ValueError('Rule outside approved whitelist')
    target = focus if focus is not None else tree
    if rule_id == 30:
        if isinstance(target, exp.Union) and target.args.get('distinct') is not False:
            return [replace_at(tree, target, exp.Intersect(this=target.this.copy(), expression=target.expression.copy(), distinct=True))]
        return entity_join(tree,target,roles,True)
    if not isinstance(target, exp.Select) or any(isinstance(e, exp.Star) for e in target.expressions):
        return []
    out = []
    group_col = role_column(roles, 'group_column')
    count_col = role_column(roles, 'count_column')
    outer_col = role_column(roles, 'outer_column')
    inner_col = role_column(roles, 'inner_column')
    def emit(changed):
        result = replace_at(tree, target, changed)
        if text(result) != text(tree):
            out.append(result)
    if rule_id in {21, 52}:
        if bool(target.args.get('distinct')) != (rule_id == 21):
            q = target.copy(); q.set('distinct', exp.Distinct() if rule_id == 21 else None); emit(q)
    elif rule_id in {31,86,180,183,113,39}:
        candidates = direct(target, exp.Count, target)
        if rule_id == 39:
            having = target.args.get('having')
            candidates = list(having.find_all(exp.Count)) if having else []
        for node in candidates:
            value = node.this
            new = None
            if rule_id == 31 and isinstance(value, exp.Distinct) and len(value.expressions) == 1:
                new = value.expressions[0].copy()
            elif rule_id == 183 and isinstance(value, exp.Column):
                new = exp.Distinct(expressions=[value.copy()])
            elif (rule_id==180 and (isinstance(value,exp.Column) or isinstance(value,exp.Distinct) and len(value.expressions)==1)) or (rule_id==39 and isinstance(value,exp.Column)):
                new = exp.Star()
            elif rule_id in {86,113} and isinstance(value, exp.Star) and count_col is not None:
                new = count_col.copy()
            if new is not None:
                out.append(replace_at(tree, node, exp.Count(this=new)))
    elif rule_id in {59,188,106,132}:
        for node in direct(target, exp.Exists, target):
            if isinstance(node.parent, exp.Not) and rule_id == 59:
                continue
            sub = node.this
            if not isinstance(sub, exp.Select): continue
            if rule_id == 59:
                out.append(replace_at(tree, node, exp.Not(this=node.copy())))
            elif rule_id == 188:
                if sub.args.get('group') or sub.args.get('having') or len(sub.expressions) != 1:
                    continue
                if isinstance(unalias(sub.expressions[0]), exp.AggFunc):
                    q = sub.copy(); q.set('expressions', [exp.Literal.number(1)])
                    out.append(replace_at(tree, sub, q))
            elif outer_col is not None and inner_col is not None:
                if outer_col.table.casefold() == inner_col.table.casefold(): continue
                if rule_id == 132 and sub.args.get('where') is None:
                    q = sub.copy()
                    if roles.get('bridge_join_sql'):
                        bridge=parse(roles['bridge_join_sql'])
                        q.set('joins',[*(j.copy() for j in q.args.get('joins',[])),*(j.copy() for j in bridge.args.get('joins',[]))])
                    q=q.where(exp.EQ(this=inner_col.copy(), expression=outer_col.copy()))
                    out.append(replace_at(tree, sub, q))
                if rule_id == 106:
                    for eq in direct(sub, exp.EQ, sub):
                        if isinstance(eq.this, exp.Column) and same(eq.this, eq.expression) and same(eq.this,inner_col):
                            out.append(replace_at(tree,eq,exp.EQ(this=inner_col.copy(),expression=outer_col.copy())))
        if rule_id==59 and not out:
            out.extend(entity_join(tree,target,roles,False))
    elif rule_id in {145,173}:
        for node in direct(target, exp.In, target):
            if not isinstance(node.parent, exp.Not) or not isinstance(node.this, exp.Column): continue
            subquery = node.args.get('query')
            sub = subquery.this if isinstance(subquery,exp.Subquery) else subquery
            if not isinstance(sub,exp.Select) or len(sub.expressions)!=1 or sub.args.get('group') or sub.args.get('limit'):continue
            source = relation(sub)
            outer_source = relation(target)
            if not isinstance(source,exp.Table) or not isinstance(outer_source,exp.Table):continue
            if source.alias_or_name.casefold()==outer_source.alias_or_name.casefold():continue
            value=unalias(sub.expressions[0])
            if not isinstance(value,exp.Column):continue
            lhs=node.this.copy(); rhs=value.copy()
            if not lhs.table: lhs.set('table',exp.to_identifier(outer_source.alias_or_name))
            if not rhs.table: rhs.set('table',exp.to_identifier(source.alias_or_name))
            q=sub.copy();q.set('expressions',[exp.Literal.number(1)]);q.set('distinct',None);q.set('order',None)
            q=q.where(exp.EQ(this=rhs,expression=lhs),append=True)
            out.append(replace_at(tree,node.parent,exp.Not(this=exp.Exists(this=q))))
    elif rule_id == 100:
        for join in target.args.get('joins',[]):
            keys=columns_from_equality(join)
            where=target.args.get('where')
            if join.args.get('side')!='LEFT' or not keys or not where:continue
            for null in direct(where,exp.Is,target):
                if isinstance(null.expression,exp.Null) and isinstance(null.this,exp.Column) and null.this.table.casefold()==join.this.alias_or_name.casefold():
                    out.append(replace_at(tree,null.this,keys[1]))
    elif rule_id in {23,190}:
        joins=target.args.get('joins',[])
        if len(joins)!=1 or joins[0].args.get('side') or not isinstance(joins[0].this,exp.Table):return []
        join=joins[0];keys=columns_from_equality(join)
        if not keys:return []
        right_alias=join.this.alias_or_name.casefold()
        # Preserve all right filters in EXISTS. Mixed OR is not moved between scopes.
        parts=[];where=target.args.get('where')
        if where:
            if isinstance(where.this,exp.Or):return []
            parts=list(where.this.flatten()) if isinstance(where.this,exp.And) else [where.this]
        related=[x for x in parts if any(c.table.casefold()==right_alias for c in local_columns(x,target))]
        other=[x for x in parts if x not in related]
        if any(any(c.table and c.table.casefold()!=right_alias for c in local_columns(x,target)) for x in related):return []
        used=[*target.expressions]+[target.args[k] for k in ('group','having','order') if target.args.get(k)]
        if any(c.table.casefold()==right_alias for x in used for c in local_columns(x,target)):return []
        q=target.copy();q.set('joins',[])
        sub=exp.select('1').from_(join.this.copy()).where(exp.and_(join.args['on'].copy(),*(x.copy() for x in related)))
        q.set('where',exp.Where(this=exp.and_(*(x.copy() for x in other),exp.Exists(this=sub))))
        emit(q)
    elif rule_id in {72,157}:
        where=target.args.get('where'); group=target.args.get('group')
        if not where or not group or isinstance(where.this,exp.Or):return []
        parts=list(where.this.flatten()) if isinstance(where.this,exp.And) else [where.this]
        sums=[unalias(e) for e in target.expressions if isinstance(unalias(e),exp.AggFunc)]
        for term in parts:
            changed=term.copy()
            if rule_id==157:
                if not any(isinstance(n,exp.AggFunc) for n in term.walk()):continue
            else:
                if not isinstance(term,(exp.GT,exp.GTE,exp.LT,exp.LTE,exp.EQ)) or not isinstance(term.this,exp.Column):continue
                matches=[a for a in sums if same(a.this,term.this)]
                if len(matches)!=1:continue
                changed.set('this',matches[0].copy())
            q=target.copy();remaining=[x.copy() for x in parts if x is not term]
            q.set('where',exp.Where(this=exp.and_(*remaining)) if remaining else None)
            q=q.having(changed,append=True);emit(q)
    elif rule_id in {25,122,172,181} and group_col is not None:
        group=target.args.get('group')
        if rule_id==25:
            if group or len(target.expressions)!=1 or not isinstance(unalias(target.expressions[0]),exp.Count):return []
            q=target.copy();q.set('expressions',[group_col.copy()]);q.set('group',exp.Group(expressions=[group_col.copy()]));q.set('order',None)
            emit(exp.select(exp.Count(this=exp.Star())).from_(q.subquery('_s1_groups')))
        elif rule_id==122 and group and any(same(e,group_col) for e in group.expressions) and len(group.expressions)>1:
            q=target.copy();q.set('group',exp.Group(expressions=[group_col.copy()]));emit(q)
        elif rule_id==172 and group and len(group.expressions)==1:
            old=group.expressions[0]
            if not same(old,group_col):
                q=target.copy()
                for col in list(q.find_all(exp.Column)):
                    if col.find_ancestor(exp.Select) is q and same(col,old): col.replace(group_col.copy())
                emit(q)
        elif rule_id==181 and not group:
            q=target.copy();q.set('expressions',[group_col.copy(),*(e.copy() for e in target.expressions)]);q.set('group',exp.Group(expressions=[group_col.copy()]));emit(q)
    return out


def entity_join(tree,target,roles,both):
    """Separate row predicates from entity quantification; no names or values invented."""
    if not isinstance(target,exp.Select) or target.args.get('group'):return []
    joins=target.args.get('joins',[]);source=relation(target);where=target.args.get('where')
    if len(joins)!=1 or not isinstance(source,exp.Table) or not isinstance(joins[0].this,exp.Table) or not where:return []
    join=joins[0];pair=columns_from_equality(join)
    if not pair or join.args.get('side') not in (None,'LEFT',''):return []
    left,right=source.alias_or_name.casefold(),join.this.alias_or_name.casefold()
    projected=[*target.expressions]+[target.args[k] for k in ('having','order') if target.args.get(k)]
    if any(c.table.casefold()!=left for e in projected for c in local_columns(e,target)):return []
    terms=list(where.this.flatten()) if isinstance(where.this,exp.And) else [where.this]
    related=[t for t in terms if any(c.table.casefold()==right for c in local_columns(t,target))]
    other=[t for t in terms if t not in related]
    if len(related)!=1:return []
    p=related[0];conditions=[]
    if both:
        values=[];col=None
        if isinstance(p,exp.In) and not p.args.get('query'):col=p.this;values=p.expressions
        elif isinstance(p,exp.Or):
            parts=list(p.flatten())
            if len(parts)==2 and all(isinstance(x,exp.EQ) and isinstance(x.expression,exp.Literal) for x in parts) and same(parts[0].this,parts[1].this):
                col=parts[0].this;values=[x.expression for x in parts]
        if not isinstance(col,exp.Column) or len(values)!=2 or not all(isinstance(v,exp.Literal) and str(v.this).casefold() in roles.get('question','').casefold() for v in values):return []
        conditions=[exp.EQ(this=col.copy(),expression=v.copy()) for v in values]
    elif isinstance(p,exp.NEQ) and isinstance(p.this,exp.Column) and isinstance(p.expression,exp.Literal) and str(p.expression.this).casefold() in roles.get('question','').casefold():
        conditions=[exp.EQ(this=p.this.copy(),expression=p.expression.copy())]
    else:return []
    queries=[exp.Exists(this=exp.select('1').from_(join.this.copy()).where(exp.and_(join.args['on'].copy(),c))) for c in conditions]
    q=target.copy();q.set('joins',[])
    condition=exp.and_(*queries) if both else exp.Not(this=queries[0])
    q.set('where',exp.Where(this=exp.and_(*(x.copy() for x in other),condition)))
    return [replace_at(tree,target,q)]


def apply_operation(sql, rule_id, roles, guard):
    """Guard has independently collected statuses, never the corpus guard object."""
    required=('structure','schema','intent','data')
    if set(guard)!=set(required) or not all(guard[k]['state'] is True and guard[k].get('evidence') for k in required):
        return []
    tree=normalize(parse(sql))
    return [text(t) for t in edit(tree,rule_id,roles)]
