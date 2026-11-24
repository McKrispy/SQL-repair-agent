"""immutable observation-derived goals and bounded physical scope binding.

No reference, database, task-ID, generated-SQL or provenance inputs belong here.
"""
import re
from sqlglot import exp
from sqlglot.errors import SqlglotError
from mp3_agent.rule_support import parse, text, header_expression, unalias, NUMBERS
from mp3_agent.semantic_facts import words
from mp3_agent.skeleton_rebuild import _parts, _prepare


def pk(schema, table):
    keys=[n for n,c in schema.columns.get(table,{}).items() if c.get('pk')]
    return keys[0] if len(keys)==1 else None


def contains(phrase, name):
    a,b=words(phrase),words(name)
    return bool(b) and any(a[i:i+len(b)]==b for i in range(len(a)-len(b)+1))


def sources(root):
    if not isinstance(root,exp.Select) or not root.args.get('from'):return None
    ts=[root.args['from'].this,*[j.this for j in root.args.get('joins',[])]]
    if any(not isinstance(t,exp.Table) for t in ts):return None
    if len({t.alias_or_name.casefold() for t in ts})!=len(ts):return None
    return {t.alias_or_name.casefold():t.name.casefold() for t in ts}


def owner(col,root,schema,outer=None):
    ss=sources(root)
    if ss is None:return None
    name=col.name.casefold()
    if col.table:
        alias=col.table.casefold()
        if alias in ss:
            table=ss[alias]
            return (table,name) if name in schema.columns.get(table,{}) else None
        # Correlation requires an explicit qualifier; a shadowed local alias never escapes.
        if outer is not None:return owner(col,outer,schema)
        return None
    hits=[(t,name) for t in ss.values() if name in schema.columns.get(t,{})]
    return hits[0] if len(hits)==1 else None


def physical(node,root,schema,outer=None):
    n=unalias(node).copy()
    for c in n.find_all(exp.Column):
        if c.is_star:return None
        p=owner(c,root,schema,outer)
        if p is None:return None
        c.set('this',exp.to_identifier(p[1],quoted=True));c.set('table',exp.to_identifier(p[0],quoted=True))
    return text(n)


def atoms(tree,schema):
    result=[]
    for s in tree.find_all(exp.Select):
        w=s.args.get('where')
        if not w:continue
        for p in _parts(w.this):
            col=None;val=None
            if isinstance(p,(exp.EQ,exp.NEQ,exp.GT,exp.GTE,exp.LT,exp.LTE,exp.Is)) and isinstance(p.this,exp.Column) and isinstance(p.expression,(exp.Literal,exp.Null)):
                col,val=p.this,p.expression
            elif isinstance(p,exp.In) and isinstance(p.this,exp.Column) and not p.args.get('query') and all(isinstance(v,exp.Literal) for v in p.expressions):col=p.this
            elif isinstance(p,exp.Not) and isinstance(p.this,exp.Is) and isinstance(p.this.this,exp.Column) and isinstance(p.this.expression,exp.Null):col=p.this.this
            if col is None:continue
            o=owner(col,s,schema)
            if o:result.append({'table':o[0],'column':o[1],'sql':text(p),'operator':type(p).__name__,
                                'values':[{'value':v.this,'string':v.is_string} for v in p.find_all(exp.Literal)]})
    return result


def root_options(question,example,schema,region=None):
    region=region if region is not None else question
    entities=[t for t in schema.tables if contains(region,t) and pk(schema,t)]
    labels=[header_expression(x) for x in example.get('columns',[])]
    ordinary=[x for x in labels if isinstance(x,exp.Column)]
    if len(entities)==1:
        t=entities[0]
        return [{'kind':'entity','table':t,'key':pk(schema,t),'certain':True}]
    domain=[]
    for h in ordinary:
        hits=[(t,c) for t,cs in schema.columns.items() for c in cs if c==h.name.casefold() and
              (not h.table or h.table.casefold()==t) and contains(region,c)]
        if len(hits)==1:domain.append(hits[0])
    domain=list(dict.fromkeys(domain))
    if len(domain)==1 and not entities:
        t,c=domain[0];return [{'kind':'domain','table':t,'key':c,'certain':True}]
    return [{'kind':'entity','table':t,'key':pk(schema,t),'certain':False} for t in sorted(entities)]


def count_goals(question,schema,config):
    """Every numeric span must bind a physical entity noun, never a field value."""
    digits=r'(?:\d+|'+'|'.join(NUMBERS)+r')'
    comps=config['semantic_language']['count_comparators']
    prefix='|'.join(re.escape(x) for x in sorted(comps,key=len,reverse=True))
    found=[]
    for m in re.finditer(r'\b(?:(?P<cmp>'+prefix+r')\s+)?(?P<n>'+digits+r')\s+(?P<tail>[a-zA-Z_][\w ]*)',question,re.I):
        tail=m['tail'];hits=[]
        for t in schema.tables:
            wt=words(t);tw=words(tail)
            if tw[:len(wt)]==wt and pk(schema,t):hits.append(t)
        if len(hits)!=1:continue
        raw=m['n'].casefold();num=int(raw) if raw.isdigit() else NUMBERS[raw]
        before=question[:m.start()].casefold()
        # Unsupported quantity modifiers do not silently become bare equality.
        if not m['cmp'] and re.search(r'\b(?:than|least|most)\s*$',before):continue
        noun_words=len(words(hits[0]));tokens=list(re.finditer(r'[a-zA-Z0-9]+',tail));end=m.start('tail')+tokens[noun_words-1].end()
        found.append({'table':hits[0],'key':pk(schema,hits[0]),'number':num,'operator':comps.get((m['cmp'] or '').casefold(),'EQ'),
                      'start':m.start(),'end':end,'evidence':question[m.start():end]})
    # The greedy tail regex above must not consume subsequent numeric goals.
    if len(found)<2:
        for m in re.finditer(r'\b('+digits+r')\b',question,re.I):
            if any(x['start']<=m.start()<x['end'] for x in found):continue
            sub=count_goals_tail(question,m.start(),schema,config)
            if sub:found.append(sub)
    return sorted({(x['start'],x['end']):x for x in found}.values(),key=lambda x:x['start'])


def count_goals_tail(q,start,schema,config):
    # Reparse a numeric suffix including its immediately preceding registered comparator.
    words_before=list(re.finditer(r'\w+',q[:start]))
    begins=[x.start() for x in words_before[-3:]]+[start]
    for begin in begins:
        piece=q[begin:];comp=None
        for c in sorted(config['semantic_language']['count_comparators'],key=len,reverse=True):
            if piece.casefold().startswith(c+' '):comp=c;break
        n=re.match((re.escape(comp)+r'\s+' if comp else '')+r'(\d+|'+'|'.join(NUMBERS)+r')\s+(.+)',piece,re.I)
        if not n:continue
        if begin+n.start(1)!=start:continue
        ts=[t for t in schema.tables if words(n[2])[:len(words(t))]==words(t) and pk(schema,t)]
        if len(ts)!=1:continue
        tok=list(re.finditer(r'[a-zA-Z0-9]+',n[2]));end=begin+n.start(2)+tok[len(words(ts[0]))-1].end()
        value=n[1].casefold()
        return {'table':ts[0],'key':pk(schema,ts[0]),'number':int(value) if value.isdigit() else NUMBERS[value],
                'operator':config['semantic_language']['count_comparators'].get(comp,'EQ'),'start':begin,'end':end,'evidence':q[begin:end]}
    return None


def phrase_region(question):
    return re.split(r'\b(?:who|that|which|with|having|never|without|do not|does not|both)\b',question,1,flags=re.I)[0]


def define(question,example,schema,raw,config):
    q=question.strip().rstrip('?.').strip();low=q.casefold();language=config['semantic_language'];events=[]
    roots=root_options(q,example,schema,phrase_region(q));raw_atoms=[];trees=[]
    for sql in raw:
        try:
            tree=parse(sql);trees.append(tree);raw_atoms.extend(atoms(tree,schema))
        except (ValueError,SqlglotError,RecursionError):continue
    raw_atoms=list({(a['table'],a['column'],a['sql']):a for a in raw_atoms}.values())
    counts=count_goals(q,schema,config)
    goals=[]
    # Question-scoped predicates: physical field words or an exact literal anchor.
    pred=[]
    for a in raw_atoms:
        values=a['values'];anchored=contains(q,a['column']) or any(v['string'] and v['value'].casefold() in low for v in values)
        if not anchored:continue
        aa=dict(a)
        if aa['operator']=='NEQ':
            node=parse('SELECT 1 WHERE '+aa['sql']).args['where'].this
            aa['sql']=text(exp.EQ(this=node.this.copy(),expression=node.expression.copy()));aa['operator']='EQ'
        pred.append(aa)
    masked=re.sub(r'\bno more than\b','at most',low)
    negative=bool(re.search(r'\b(?:never|without)\b|\b(?:do|does) not\b|\bnot\b.+\bany\b|\bno\b',masked))
    verb=any(re.search(r'\b'+re.escape(v)+r'\b',low) for v in language['verbs'])
    relations=[t for t in schema.tables if contains(q,t) and all(t!=r['table'] for r in roots)]
    if negative and (verb or re.search(r'\bwithout\b|\bno\b',masked)):
        candidates=list(dict.fromkeys(relations+[p['table'] for p in pred if all(p['table']!=r['table'] for r in roots)]))
        # Domain anti-existence may exclude another row of the same physical table.
        if any(r['kind']=='domain' for r in roots):candidates+=list(dict.fromkeys(p['table'] for p in pred if p['table'] not in candidates))
        if roots and candidates:goals.append({'id':'relation_absence','family':'R1','roots':roots,'relations':candidates,'predicates':pred})
        else:events.append('absence_root_or_relation_unbound')
    both=list(re.finditer(r'\bboth\b',low))
    if len(both)==1 and not negative:
        tail=q[both[0].end():];quote=re.findall(r"['\"]([^'\"]+)['\"]",tail)
        parts=re.split(r'\band\b',re.sub(r"(['\"])(.*?)\1",lambda m:m[0].replace(' and ',' __quoted_and__ '),tail),flags=re.I)
        if len(parts)==2:
            vals=quote if len(quote)==2 else [p.strip(' \"\'.,?').replace('__quoted_and__','and') for p in parts]
            pairs=[]
            fields={(p['table'],p['column']) for p in pred if p['values']}
            for t,c in sorted(fields):
                branch=[]
                for value in vals:
                    hits=[v['value'] for a in raw_atoms if (a['table'],a['column'])==(t,c) for v in a['values'] if v['string'] and v['value'].casefold()==value.casefold()]
                    branch.append({'table':t,'column':c,'operator':'EQ','value':hits[0] if hits else value,'string':True,'known':bool(hits)})
                if len(branch)==2:pairs.append(branch)
            # Two directly anchored simple predicates can be on different tables.
            if not pairs:
                branches=[]
                for part in parts:
                    hits=[a for a in pred if any(v['value'].casefold() in part.casefold() for v in a['values'])]
                    if len(hits)==1:branches.append(hits[0])
                if len(branches)==2:pairs.append(branches)
            if roots and pairs:goals.append({'id':'relation_both','family':'R2','roots':roots,'branches':pairs[:config['binding_beam']]})
            else:events.append('both_value_or_root_unbound')
        else:events.append('both_boolean_scope_unsupported')
    group=re.search(r'\b(?:'+ '|'.join(re.escape(x) for x in language['groups'])+r')\s+(?:the\s+)?(.+)',q,re.I)
    shared=re.search(r'\bshared by\b',low)
    if group and not re.search(r'\b(rows|records|entries|occurrences)\b',low):
        gr=root_options(q,example,schema,group[1]);countmatch=re.search(r'\b(?:how many|number of|count of)\s+(?:the\s+)?(.+)',q,re.I)
        ts=[t for t in schema.tables if countmatch and words(countmatch[1])[:len(words(t))]==words(t) and pk(schema,t)]
        if gr and len(ts)==1:goals.append({'id':'count_group','family':'R3','roots':gr,'count_table':ts[0],'count_key':pk(schema,ts[0]),'quantity':None,'full_frame':True})
    if len(counts)==1 and not group and not negative and not both:
        region=q[:counts[0]['start']];gr=root_options(q,example,schema,region)
        domain=[r for r in gr if r['kind']=='domain']
        if domain and (shared or re.search(r'\b(?:with|that have)\b',region,re.I)) and complete_frame(q,counts,config,schema):
            goals.append({'id':'count_group','family':'R3','roots':domain,'count_table':counts[0]['table'],'count_key':counts[0]['key'],'quantity':counts[0],'full_frame':True})
    if len(counts)==2 and not negative and not both:
        between=q[counts[0]['end']:counts[1]['start']].strip()
        op=re.fullmatch(r'(and|or)\s*(?:'+ '|'.join(re.escape(x) for x in language['count_links'])+r')?\s*',between,re.I)
        if op and roots and complete_frame(q,counts,config,schema):goals.append({'id':'two_counts','family':'R4','roots':roots,'counts':counts,'join':op[1].upper(),'full_frame':True})
        else:events.append('two_counts_frame_unconsumed')
    elif len(counts)>config['max_semantic_count_goals']:events.append('too_many_count_goals')
    if any(re.search(r'\b'+re.escape(m)+r'\b',low) for m in language['measure']):
        ms=[]
        for tree in trees:
            if not isinstance(tree,exp.Select):continue
            aggs=[n for e in tree.expressions for n in e.find_all(exp.AggFunc)]
            if len(aggs)!=1 or not isinstance(aggs[0],(exp.Sum,exp.Avg)):continue
            arg=aggs[0].this;arg=arg.this if isinstance(arg,exp.Cast) else arg
            if not isinstance(arg,exp.Column):continue
            p=owner(arg,tree,schema)
            if p and pk(schema,p[0]) and contains(q,p[1]) and contains(q,p[0]):ms.append((p[0],p[1],type(aggs[0]).__name__))
        for t,c,op in sorted(set(ms))[:config['max_semantic_measure_goals']]:goals.append({'id':'measure_grain','family':'R5','table':t,'column':c,'operation':op})
    relations_goals=[g for g in goals if g['family'] in ('R1','R2')]
    if len(relations_goals)>config['max_semantic_relation_goals']:goals=[g for g in goals if g not in relations_goals];events.append('too_many_relation_goals')
    return {'goals':goals,'unsupported':events,'raw_atoms':raw_atoms,'question':q}


def complete_frame(q,counts,config,schema):
    """Only registered connective/output/physical-name tokens may remain."""
    remaining=q
    for c in reversed(counts):remaining=remaining[:c['start']]+' '+remaining[c['end']:]
    phrases=config['semantic_language']['projection']+config['semantic_language']['count_links']+['shared by','that are','and','or','the','all','of','name','names','id','ids','detail','details']
    for phrase in sorted(phrases,key=len,reverse=True):remaining=re.sub(r'\b'+re.escape(phrase)+r'\b',' ',remaining,flags=re.I)
    allowed=set()
    for t,cs in schema.columns.items():allowed.update(words(t));[allowed.update(words(c)) for c in cs]
    return all(w in allowed for w in words(remaining))


def feedback_class(f):
    if f is None:return 'unseen'
    if not f.get('ok'):
        error=f.get('error') or {};message=error.get('message','') if isinstance(error,dict) else str(error)
        kind=error.get('type','') if isinstance(error,dict) else ''
        if any(message.startswith(x) for x in ('no such column:','no such table:','ambiguous column name:')):return 'binding_error'
        return 'invalid_sql' if kind=='InvalidSQL' else 'resource_or_other_error'
    if f.get('matches_example'):
        if not f.get('truncated') and f.get('row_count')==0:return 'pass_weak_empty'
        rows=f.get('rows')
        if not f.get('truncated') and f.get('row_count')==1 and rows and rows[0] and all(x is None for x in rows[0]):return 'pass_weak_null'
        return 'pass_other'
    return 'mismatch_truncated' if f.get('truncated') else 'mismatch_empty' if f.get('row_count')==0 else 'mismatch_rows'
