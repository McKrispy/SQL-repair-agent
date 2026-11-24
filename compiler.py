"""Bind public output expressions and recombine SQL clauses over the current paid schema."""
from collections import Counter, defaultdict, deque
from itertools import product
import re
from sqlglot import exp,parse_one
from sqlglot.errors import SqlglotError
from mp3_agent.rule_support import header_expression
from mp3_agent.safety import sql_key
from pattern import logits,grade

def ident(x):
    return '"'+x.replace('"','""')+'"'

def words(x):
    return {w.rstrip("s") for w in re.findall(r"[a-z]+",re.sub(r"([a-z])([A-Z])",r"\1 \2",x).lower())}

def spell(e):
    return e.sql(dialect="sqlite",comments=False)

class Catalog:
    def __init__(self,schema,question,roots):
        self.q=question.lower()
        self.words=words(question)
        self.columns={}
        self.owners=defaultdict(list)
        self.edges=defaultdict(list)
        self.votes=Counter()
        self.bind_votes=Counter()
        for t in schema:
            table=t["name"].lower()
            self.columns[table]={c["name"].lower() for c in t["columns"]}
            for col in self.columns[table]:
                self.owners[col].append(table)
            for fk in t["foreign_keys"]:
                if not fk["to"]:
                    continue
                other=fk["table"].lower()
                a,b=fk["from"].lower(),fk["to"].lower()
                self.edges[table].append((other,a,b))
                self.edges[other].append((table,b,a))
        for root in roots:
            self.votes.update({t.name.lower() for t in root.find_all(exp.Table)})
            for s in root.find_all(exp.Select):
                aliases=self.aliases(s)
                for e in s.expressions:
                    for c in e.find_all(exp.Column):
                        t=aliases.get(c.table.lower())
                        if t:
                            self.bind_votes[(c.name.lower(),t)]+=1
    @staticmethod
    def aliases(s):
        f=s.args.get("from")
        ts=([f.this] if f else [])+[j.this for j in s.args.get("joins",[])]
        return {t.alias_or_name.lower():t.name.lower() for t in ts if isinstance(t,exp.Table)}

    def affinity(self,t):
        tw=words(t)
        return min(2,self.votes[t])*.28 + (1.6 if tw and tw<=self.words else .4*len(tw & self.words))

    def bind(self,expr,aliases=None):
        cs=list(expr.find_all(exp.Column))
        options=[]
        for c in cs:
            name=c.name.lower()
            possible=self.owners.get(name,[])
            if not possible:
                return []
            resolved=(aliases or {}).get(c.table.lower())
            inside=set((aliases or {}).values())
            if resolved in possible:
                possible=[resolved]
            elif inside.intersection(possible):
                possible=[t for t in possible if t in inside]
            options.append(sorted(possible,key=lambda t:(-self.bind_votes[name,t],-self.affinity(t),t))[:3])
        rows=[]
        for targets in product(*options):
            bound=expr.copy()
            for c,t in zip(bound.find_all(exp.Column),targets):
                c.set("table",exp.to_identifier(t,quoted=True))
                c.set("this",exp.to_identifier(c.name.lower(),quoted=True))
            rows.append((spell(bound),tuple(sorted(set(targets)))))
            if len(rows)>=8:
                break
        return rows

    def join(self, targets):
        targets=tuple(sorted(set(targets)))
        if not targets:
            return None
        built={targets[0]}
        sql=ident(targets[0])
        while set(targets)-built:
            todo=deque((t,[]) for t in sorted(built))
            visited=set(built)
            path=None
            while todo:
                current,steps=todo.popleft()
                if current in targets and current not in built:
                    path=steps
                    break
                if len(steps)>=3:
                    continue
                for other,a,b in sorted(self.edges[current]):
                    if other not in visited:
                        visited.add(other)
                        todo.append((other,steps+[(current,other,a,b)]))
            if not path:
                return None
            for a,b,ac,bc in path:
                sql+=f" JOIN {ident(b)} ON {ident(a)}.{ident(ac)} = {ident(b)}.{ident(bc)}"
                built.add(b)
            if len(built)>4:
                return None
        return sql,built

def roots_from(obs):
    hints=next((e["observation"] for e in obs["history"] if e["tool"]=="repair_hint"),[])
    roots=[]
    for sql in dict.fromkeys([obs["issue_sql"],*hints,*[e["sql"] for e in obs["history"] if e["tool"]=="check"]]):
        try:
            root=parse_one(sql,read="sqlite")
        except (ValueError,SqlglotError,RecursionError):
            continue
        if root:
            roots.append(root)
    return roots

def clause_pool(roots,catalog):
    filters,sorts,groups=Counter(),Counter(),Counter()
    for root in roots:
        vf,vs,vg=set(),set(),set()
        for s in root.find_all(exp.Select):
            aliases=catalog.aliases(s)
            where=s.args.get("where")
            if where:
                parts=[where.this]
                if isinstance(where.this,(exp.And,exp.Or)):
                    parts+=list(where.this.flatten())
                for expr in parts:
                    if expr.find(exp.Select) or not expr.find(exp.Column) or not expr.find(exp.Literal):
                        continue
                    vf.update(catalog.bind(expr,aliases))
                    if isinstance(expr,(exp.EQ,exp.NEQ)) and isinstance(expr.this,exp.Column) and isinstance(expr.expression,exp.Literal) and expr.expression.is_string:
                        v=expr.copy()
                        v.set("this",exp.Lower(this=v.this.copy()))
                        v.set("expression",exp.Literal.string(expr.expression.this.lower()))
                        vf.update(catalog.bind(v,aliases))
            for item in s.args.get("order",exp.Order()).expressions:
                if item.this.find(exp.Select) or not (item.this.find(exp.Column) or item.this.find(exp.AggFunc)):
                    continue
                vs.update(catalog.bind(item.this,aliases))
            for item in s.args.get("group",exp.Group()).expressions:
                if isinstance(item,exp.Column):
                    vg.update(catalog.bind(item,aliases))
        filters.update(vf);sorts.update(vs);groups.update(vg)
    return filters,sorts,groups

def emit(obs,model):
    question=obs["task"]["question"]
    q=question.lower()
    roots=roots_from(obs)
    cat=Catalog(obs["history"][0]["observation"],question,roots)
    score=logits(model,question)
    filters,sorts,group_votes=clause_pool(roots,cat)
    headers=[header_expression(h) for h in obs["example"]["columns"]]
    if not headers or len(headers)>7 or any(h is None for h in headers):
        return []
    options=[cat.bind(h) for h in headers]
    if any(not xs for xs in options):
        return []
    top=lambda items,k:sorted(items,key=lambda x:(-items[x],x))[:k]
    flist=[("",()),*top(filters,10)]
    slist=top(sorts,4) if obs["ordered"] else []
    limit=re.search(r"\b(?:top|first|last|bottom)\s+(\d+)\b",q)
    k=min(100,int(limit[1])) if limit else 1
    count_cue=bool(re.search(r"\b(most|least|fewest|greatest number|most common|most often|largest number|highest number|lowest number)\b",q))
    if obs["ordered"] and (count_cue or score["order_count"]>0):
        slist.append(("COUNT(*)",()))
    direction="DESC" if score["desc"]>0 or re.search(r"most|highest|largest|descending|decreasing|greatest|latest",q) else "ASC"
    if re.search(r"fewest|least|lowest|smallest|ascending|earliest",q):
        direction="ASC"
    threshold=re.search(r"\b(at least|more than|greater than|over|less than|fewer than|exactly)\s+(\d+|one|two|three|four|five)\b",q)
    clauses={}
    def remember(sql,value):
        if sql not in clauses or value>clauses[sql]:
            clauses[sql]=value
    for i,proj in enumerate(product(*options)):
        if i>=18:
            break
        cols=[p[0] for p in proj]
        nonagg=[p for p,h in zip(proj,headers) if isinstance(h,exp.Column)]
        projected=set(t for _,ts in proj for t in ts)
        anchors=[()] if projected else [(t,) for t in sorted(cat.columns,key=lambda t:(-cat.affinity(t),t))[:4]]
        for condition,ft in flist:
            cond_vote=filters[condition,ft] if condition else 0
            for ordering,ot in [("",()),*slist,*([nonagg[0]] if obs["ordered"] and nonagg else [])]:
                for anchor in anchors:
                    tables=projected|set(ft)|set(ot)|set(anchor)
                    joined=cat.join(tables)
                    if not joined:
                        continue
                    froms,physical=joined
                    if not obs["ordered"] and ordering:
                        continue
                    # A count grouped by the displayed dimension and COUNT top-k often have no ORDER clue in the draft.
                    gcols=", ".join(p[0] for p in nonagg)
                    needs_group=bool(nonagg and (len(nonagg)<len(proj) or ordering=="COUNT(*)"))
                    gopts=[False,True] if gcols and (needs_group or threshold or score["group"]>-1) else [False]
                    for grouped in gopts:
                        flags=set()
                        if grouped:flags.add("group")
                        if condition:flags.add("where")
                        if ordering=="COUNT(*)":flags.add("order_count")
                        if ordering and direction=="DESC":flags.add("desc")
                        limit_on=bool(ordering and (limit or count_cue or score["limit"]>-.5) and obs["example"]["row_count"]<=k)
                        if limit_on:flags.add("limit")
                        hv=""
                        if grouped and threshold and (score["having"]>-.8 or re.search(r"number|count|times|at least",q)):
                            number={"one":1,"two":2,"three":3,"four":4,"five":5}.get(threshold[2],threshold[2])
                            op={"at least":">=","exactly":"=","less than":"<","fewer than":"<"}.get(threshold[1],">")
                            hv=f" HAVING COUNT(*) {op} {number}"
                            flags.add("having")
                        distinct=score["distinct"]>0 and not grouped and len(nonagg)==len(proj)
                        if distinct:flags.add("distinct")
                        sql="SELECT "+("DISTINCT " if distinct else "")+", ".join(cols)+" FROM "+froms
                        if condition:sql+=" WHERE "+condition
                        if grouped:sql+=" GROUP BY "+gcols+hv
                        if ordering:sql+=" ORDER BY "+ordering+" "+direction
                        if limit_on:sql+=" LIMIT "+str(k)
                        rank=grade(score,flags)+sum(cat.affinity(t) for t in physical)-1.2*(len(physical)-1)
                        rank+=min(2.4,cond_vote*.6) + (1.3 if condition and "LOWER(" not in condition.upper() else 0)
                        rank+=int(bool(ordering)==obs["ordered"])*3+int(grouped==needs_group)*1.4
                        rank+=sum(min(1.5,cat.bind_votes[h.name.lower(),p[1][0]]*.45) for p,h in zip(proj,headers) if isinstance(h,exp.Column) and p[1])
                        if ordering:rank+=min(1.5,sorts[ordering,ot]*.4)
                        if ordering=="COUNT(*)" and count_cue:rank+=2.5
                        remember(sql,rank)
        # Set semantics operate on the selected attributes; each arm gets its own join graph.
        if not nonagg or len(nonagg)!=len(proj) or not projected:
            continue
        base=cat.join(projected)
        if not base:continue
        select="SELECT "+", ".join(cols)+" FROM "
        neg=bool(re.search(r"\b(no|never|without|not|don't|doesn't|except)\b",q))
        both=bool(re.search(r"\bboth\b",q))
        if neg or both:
            atoms=[(f,ts) for f,ts in top(filters,12) if "LOWER(" not in f.upper() and not any(x in f.upper() for x in (" AND "," OR "))]
            for ix,(f,ts) in enumerate(atoms):
                right=cat.join(projected|set(ts))
                if not right:continue
                if neg:
                    pos=re.sub(r"\bNOT\s+IN\b","IN",f.replace(" <> "," = ").replace(" != "," = ").replace(" NOT LIKE "," LIKE "),flags=re.I)
                    remember(select+base[0]+" EXCEPT "+select+right[0]+" WHERE "+pos,6+min(2,filters[f,ts])+.6*score["except"])
                elif both:
                    for g,gs in atoms[ix+1:]:
                        left=cat.join(projected|set(gs))
                        if left and g!=f:
                            remember(select+left[0]+" WHERE "+g+" INTERSECT "+select+right[0]+" WHERE "+f,7+.6*score["intersect"])
    return sorted(clauses,key=lambda s:(-clauses[s],len(s),s))[:100]

