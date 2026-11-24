"""Five finite constructors. AST edits only; no benchmark execution."""
from collections import deque
from itertools import product
from sqlglot import exp
from sqlglot.errors import SqlglotError
from mp3_agent.rule_support import parse,text,unalias,header_expression
from mp3_agent.skeleton_rebuild import _parts,_prepare,_supported
from mp3_agent.semantic_search_specs import owner,sources,pk


def col(table,name,schema,alias):
    return exp.column(schema.columns[table][name]['name'],table=alias,quoted=True)


def table(t,schema,alias):
    return exp.Table(this=exp.to_identifier(schema.tables[t]['name'],quoted=True),alias=exp.TableAlias(this=exp.to_identifier(alias,quoted=True)))


def conjunction(parts):
    return exp.and_(*[p.copy() for p in parts]) if parts else None


def paths(start,end,schema,config,events):
    if start==end:return [[]]
    pending=deque([(start,[],{start})]);found=[];states=0
    while pending and states<config['skeleton_tree_states'] and len(found)<config['skeleton_max_trees']:
        t,path,seen=pending.popleft();states+=1
        if len(path)>=config['skeleton_max_edges']:continue
        for edge in sorted(schema.edges.get(t,[])):
            if edge[2] in seen:continue
            nxt=path+[edge]
            if edge[2]==end:found.append(nxt)
            else:pending.append((edge[2],nxt,seen|{edge[2]}))
            if len(found)>=config['skeleton_max_trees']:break
    if pending:events.append({'reason':'path_search_cap','states':states,'pending':len(pending),'from':start,'to':end})
    return found


def map_node(node,parent,schema,mapping):
    n=node.copy()
    if list(n.find_all(exp.Select)):return None
    if isinstance(node,exp.Column):
        p=owner(node,parent,schema)
        return col(p[0],p[1],schema,mapping[p[0]]) if p is not None and p[0] in mapping else None
    originals=list(node.find_all(exp.Column));copies=list(n.find_all(exp.Column))
    for a,b in zip(originals,copies,strict=True):
        p=owner(a,parent,schema)
        if p is None or p[0] not in mapping:return None
        b.replace(col(p[0],p[1],schema,mapping[p[0]]))
    return n


def projection(parent,root,example,schema,config,goal):
    t=root['table'];mapping={t:'r0'};out=[]
    for e in parent.expressions:
        if isinstance(unalias(e),exp.Column) and not e.is_star:
            n=map_node(e,parent,schema,mapping)
            if n is None:break
            out.append(n)
        elif isinstance(unalias(e),exp.Count) and goal['family']=='R3':
            out.append(e.copy())
        elif isinstance(unalias(e),exp.Count) and root['kind']=='entity':
            # A question-derived explicit count must be bound by the caller.
            return None
        else:break
    else:
        if out and len(out)==len(example.get('columns',[])):return out
    out=[]
    for label in example.get('columns',[]):
        h=header_expression(label)
        if not isinstance(h,exp.Column) or h.name.casefold() not in schema.columns[t] or h.table and h.table.casefold()!=t:return None
        hits=[u for u,cs in schema.columns.items() if h.name.casefold() in cs]
        if len(hits)!=1 and not root.get('certain'):return None
        out.append(col(t,h.name.casefold(),schema,'r0'))
    return out or None


def base(parent,root,example,schema,config,goal,events):
    if not isinstance(parent,exp.Select) or parent.args.get('with') or list(parent.find_all(exp.Window)):return None
    try:parent=_prepare(parent,schema)
    except (ValueError,KeyError,SqlglotError):return None
    if any(j.args.get('side') or j.args.get('kind') not in (None,'INNER') for j in parent.args.get('joins',[])):return None
    projections=projection(parent,root,example,schema,config,goal)
    if not projections:return None
    q=exp.select(*projections).from_(table(root['table'],schema,'r0'))
    if root['kind']=='domain':q=q.distinct()
    elif parent.args.get('distinct'):
        # Entity identity is not DISTINCT display name. Do not carry a name collapse.
        events.append({'reason':'entity_display_distinct_not_copied'})
    kept=[];removed=[];w=parent.args.get('where')
    allowed=set(goal.get('relations',[]))
    if goal['family']=='R2':allowed.update(p['table'] for bs in goal['branches'] for p in bs)
    if goal['family']=='R4':allowed.update(c['table'] for c in goal['counts'])
    for p in _parts(w.this if w else None):
        mapped=map_node(p,parent,schema,{root['table']:'r0'})
        ps=list(p.find_all(exp.Select));os={owner(c,parent,schema) for c in p.find_all(exp.Column) if c.find_ancestor(exp.Select) is parent}
        if mapped is not None:
            # A root-row negative/both predicate belongs to a domain relation, not an independent filter.
            target_domain=root['kind']=='domain' and goal['family'] in ('R1','R2') and any(
                a['column']==c.name.casefold() and a['table']==root['table'] for c in p.find_all(exp.Column)
                for a in (goal.get('predicates',[]) if goal['family']=='R1' else [x for bs in goal['branches'] for x in bs]))
            if not target_domain:kept.append(mapped);continue
        # Only a located relationship branch may be removed. Other predicates block rewriting.
        tables={t.name.casefold() for t in p.find_all(exp.Table)}|{x[0] for x in os if x}
        has_target=bool(tables & allowed) or root['kind']=='domain' and root['table'] in tables
        if goal['family']=='R4' and goal.get('full_frame') and (list(p.find_all(exp.Count)) or list(p.find_all(exp.Subquery))):has_target=True
        if not has_target or isinstance(p,exp.Or) and goal['family']!='R4':return None
        removed.append(text(p))
    if w and not kept and not removed:return None
    if kept:q=q.where(conjunction(kept))
    for clause in ('group','having','order','limit','offset'):
        node=parent.args.get(clause)
        if node is None:continue
        if goal['family']=='R4' and clause in ('group','having') and goal.get('full_frame'):
            # A complete independent-count frame replaces its old join-count machinery.
            if clause=='having' and any(not isinstance(p,(exp.EQ,exp.GT,exp.GTE,exp.LT,exp.LTE,exp.And,exp.Or,exp.Paren)) for p in [node.this]):return None
            continue
        n=map_node(node,parent,schema,{root['table']:'r0'})
        if n is None or list(n.find_all(exp.AggFunc)):return None
        q.set(clause,n)
    return q


def branch(root,target,path,schema,number,predicate=None,count=False):
    # Every branch receives stable aliases independent of parent spelling.
    if root['kind']=='domain':
        roles=[root['table']]+[e[2] for e in path]
        if target!=root['table'] and not path:return None
        aliases=[f'b{number}_{i}' for i in range(len(roles))]
        inner=exp.select(exp.Literal.number(1)).from_(table(roles[0],schema,aliases[0]))
        corr=exp.Is(this=col(root['table'],root['key'],schema,aliases[0]),expression=col(root['table'],root['key'],schema,'r0'))
        begin=0
    else:
        if not path:return None
        roles=[e[2] for e in path];aliases=[f'b{number}_{i}' for i in range(len(roles))]
        inner=exp.select(exp.Literal.number(1)).from_(table(roles[0],schema,aliases[0]))
        first=path[0];corr=exp.EQ(this=col(first[0],first[1],schema,'r0'),expression=col(first[2],first[3],schema,aliases[0]));begin=1
    for i,edge in enumerate(path[begin:],begin):
        left=aliases[i-1] if root['kind']=='domain' else aliases[i-1]
        right=aliases[i] if root['kind']=='entity' else aliases[i+1]
        # Domain role list includes its inner copy of the root.
        if root['kind']=='domain':left=aliases[i]
        condition=exp.EQ(this=col(edge[0],edge[1],schema,left),expression=col(edge[2],edge[3],schema,right))
        inner=inner.join(table(edge[2],schema,right),on=condition,join_type='INNER')
    if predicate:
        alias=aliases[-1]
        if 'sql' in predicate:
            p=parse('SELECT 1 WHERE '+predicate['sql']).args['where'].this
            for c in list(p.find_all(exp.Column)):c.replace(col(target,predicate['column'],schema,alias))
        else:p=getattr(exp,predicate['operator'])(this=col(target,predicate['column'],schema,alias),expression=exp.Literal.string(predicate['value']) if predicate['string'] else exp.Literal.number(predicate['value']))
        corr=exp.and_(corr,p)
    inner=inner.where(corr)
    if count:
        key=pk(schema,target)
        if key is None:return None
        inner.set('expressions',[exp.Count(this=exp.Distinct(expressions=[col(target,key,schema,aliases[-1])]))])
        return exp.Subquery(this=inner)
    return exp.Exists(this=inner)


def within(tree,config):
    sels=list(tree.find_all(exp.Select))
    depth=max((sum(isinstance(a,exp.Select) for a in ancestors(s)) for s in sels),default=0)
    tables=list(tree.find_all(exp.Table))
    # Each relational comparison between roles is one edge use; do not merge occurrences across branches.
    edges=sum(isinstance(n,(exp.EQ,exp.Is)) and isinstance(n.this,exp.Column) and isinstance(n.expression,exp.Column) and n.this.table!=n.expression.table for n in tree.walk())
    return depth<=config['max_semantic_sql_depth'] and len(tables)<=config['skeleton_max_tables'] and edges<=config['skeleton_max_edges']


def ancestors(n):
    while n.parent is not None:n=n.parent;yield n


def measure(parent,goal,schema,config,events):
    if _supported(parent):return
    try:q=_prepare(parent,schema)
    except (ValueError,KeyError,SqlglotError):return
    ss=sources(q);root=goal['table'];aliases=[a for a,t in ss.items() if t==root]
    if len(aliases)!=1 or len(ss)<2:return
    ra=aliases[0];edges=[];predicates=[]
    for j in q.args.get('joins',[]):
        if j.args.get('on') is None:return
        for p in _parts(j.args['on']):
            if isinstance(p,exp.EQ) and isinstance(p.this,exp.Column) and isinstance(p.expression,exp.Column):
                a=owner(p.this,q,schema);b=owner(p.expression,q,schema)
                if a is None or b is None or (a[0],a[1],b[0],b[1]) not in schema.edges.get(a[0],[]):return
                edges.append(p)
            else:predicates.append(p)
    cuts=[p for p in edges if any(c.table.casefold()==ra for c in p.find_all(exp.Column))]
    if len(cuts)!=1:return
    outer={root:'r0'}
    projections=[map_node(e,q,schema,outer) for e in q.expressions]
    if any(e is None for e in projections):return
    aggs=[a for e in projections for a in e.find_all(exp.AggFunc)]
    if len(aggs)!=1 or type(aggs[0]).__name__!=goal['operation']:return
    arg=aggs[0].this;arg=arg.this if isinstance(arg,exp.Cast) else arg
    if not isinstance(arg,exp.Column) or arg.name.casefold()!=goal['column']:return
    if any(not isinstance(unalias(e),(exp.Column,exp.Sum,exp.Avg)) for e in projections):return
    others=[t for t in ss.values() if t!=root];mapping={root:'r0',**{t:f'b0_{i}' for i,t in enumerate(others)}}
    # Remaining declared graph must be connected after removing the root.
    graph={t:set() for t in others}
    for p in edges:
        a,b=[owner(c,q,schema)[0] for c in (p.this,p.expression)]
        if a in graph and b in graph:graph[a].add(b);graph[b].add(a)
    seen=set();pending=[others[0]]
    while pending:
        t=pending.pop()
        if t in seen:continue
        seen.add(t);pending.extend(graph[t]-seen)
    if len(seen)!=len(others):return
    inner=exp.select(exp.Literal.number(1)).from_(table(others[0],schema,mapping[others[0]]))
    # Build a connected tree and leave additional FK equalities in WHERE.
    attached={others[0]};remaining=set(others[1:]);inner_parts=[]
    while remaining:
        candidate=next(((t,p) for t in sorted(remaining) for p in edges if {owner(c,q,schema)[0] for c in (p.this,p.expression)} & attached and t in {owner(c,q,schema)[0] for c in (p.this,p.expression)}),None)
        if candidate is None:return
        t,p=candidate;inner=inner.join(table(t,schema,mapping[t]),on=map_node(p,q,schema,mapping),join_type='INNER');attached.add(t);remaining.remove(t)
    rootparts=[];w=q.args.get('where');predicates+=_parts(w.this if w else None)
    for p in edges+predicates:
        roles={owner(c,q,schema) for c in p.find_all(exp.Column)}
        if None in roles:return
        rootonly=roles and all(t==root for t,c in roles)
        n=map_node(p,q,schema,mapping)
        if n is None:return
        (rootparts if rootonly else inner_parts).append(n)
    inner=inner.where(conjunction(inner_parts));out=exp.select(*projections).from_(table(root,schema,'r0')).where(conjunction(rootparts+[exp.Exists(this=inner)]))
    for clause in ('group','having','order','limit','offset'):
        if q.args.get(clause) is not None:
            n=map_node(q.args[clause],q,schema,outer)
            if n is None:return
            out.set(clause,n)
    if within(out,config):yield out,{'measure_grain_assumption':True,'root':root,'cut':text(cuts[0])}


def proposals(parent,goal,example,schema,config,events):
    try:
        tree=parse(parent['sql'])
        if goal['family']=='R5':
            yield from measure(tree,goal,schema,config,events);return
        for root in goal['roots'][:config['binding_beam']]:
            baseq=base(tree,root,example,schema,config,goal,events)
            if baseq is None:continue
            family=goal['family'];plans=[]
            if family=='R1':
                for target in goal['relations']:
                    predicates=[p for p in goal['predicates'] if p['table']==target]
                    for path in paths(root['table'],target,schema,config,events):
                        for p in predicates or [None]:
                            e=branch(root,target,path,schema,0,p)
                            if e is not None:plans.append(([exp.Not(this=e)],{'paths':[path],'predicate':p}))
            elif family=='R2':
                for pair in goal['branches']:
                    choices=[paths(root['table'],p['table'],schema,config,events) for p in pair]
                    for ps in product(*choices):
                        es=[branch(root,p['table'],path,schema,i,p) for i,(p,path) in enumerate(zip(pair,ps))]
                        if all(e is not None for e in es):plans.append((es,{'paths':list(ps),'value_binding_unknown':any(not p.get('known',True) for p in pair)}))
                        if len(plans)>=config['skeleton_max_trees']:break
            elif family=='R4':
                choices=[paths(root['table'],c['table'],schema,config,events) for c in goal['counts']]
                for ps in product(*choices):
                    es=[]
                    for i,(c,path) in enumerate(zip(goal['counts'],ps)):
                        sub=branch(root,c['table'],path,schema,i,count=True)
                        if sub is None:break
                        es.append(getattr(exp,c['operator'])(this=sub,expression=exp.Literal.number(c['number'])))
                    if len(es)==2:plans.append(([getattr(exp,goal['join'].title())(this=es[0],expression=es[1])],{'paths':list(ps)}))
                    if len(plans)>=config['skeleton_max_trees']:break
            elif family=='R3':
                target=goal['count_table']
                for path in paths(root['table'],target,schema,config,events):
                    if root['kind']=='entity':
                        sub=branch(root,target,path,schema,0,count=True)
                        if sub is None:continue
                        out=baseq.copy();exprs=[];replaced=False
                        for e in out.expressions:
                            if isinstance(unalias(e),exp.Count):exprs.append(exp.alias_(sub.copy(),e.alias,quoted=True) if e.alias else sub.copy());replaced=True
                            else:exprs.append(e)
                        if not replaced:exprs.append(sub)
                        out.set('expressions',exprs);out.set('group',None);out.set('having',None)
                        if within(out,config):yield out,{'root':root,'paths':[path],'zero_inclusion_unknown':True}
                    else:
                        mapping={root['table']:'r0'};out=baseq.copy()
                        for i,edge in enumerate(path):
                            alias=f'b0_{i}';mapping[edge[2]]=alias
                            out=out.join(table(edge[2],schema,alias),on=exp.EQ(this=col(edge[0],edge[1],schema,mapping[edge[0]]),expression=col(edge[2],edge[3],schema,alias)),join_type='INNER')
                        cnt=exp.Count(this=exp.Distinct(expressions=[col(target,goal['count_key'],schema,mapping[target])]))
                        out.set('distinct',None);out=out.group_by(col(root['table'],root['key'],schema,'r0'))
                        if goal['quantity']:
                            c=goal['quantity'];out=out.having(getattr(exp,c['operator'])(this=cnt,expression=exp.Literal.number(c['number'])))
                        else:
                            exprs=[exp.alias_(cnt.copy(),e.alias,quoted=True) if e.alias else cnt.copy() if isinstance(unalias(e),exp.Count) else e for e in out.expressions]
                            if not any(isinstance(unalias(e),exp.Count) for e in out.expressions):exprs.append(cnt)
                            out.set('expressions',exprs)
                        if within(out,config):yield out,{'root':root,'paths':[path]}
            for conditions,evidence in plans:
                q=baseq.copy();w=q.args.get('where');parts=([w.this] if w else [])+conditions;q.set('where',exp.Where(this=conjunction(parts)))
                if within(q,config):yield q,{'root':root,**evidence}
                else:events.append({'reason':'semantic_sql_role_edge_or_depth_cap','family':family})
    except (ValueError,KeyError,SqlglotError,RecursionError) as error:
        events.append({'reason':'unsupported_semantic_parent','error':str(error)})
