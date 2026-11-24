"""approved question/schema specifications and two one-level rescue families."""
import re
from sqlglot import exp
from sqlglot.errors import SqlglotError
from mp3_agent.rule_support import parse,text,header_expression,quantity,unalias
from mp3_agent.semantic_facts import words
from mp3_agent.skeleton_rebuild import _parts,_physical_owner,_sources


def grain_spec(question,example,schema,config):
    target=quantity(question)
    if not target or len(example.get('columns',[]))!=1:return None
    header=header_expression(example['columns'][0])
    if not isinstance(header,exp.Column) or not header.name.isascii():return None
    tables=[t for t in schema.tables if words(t)==words(target['noun']) and header.name.casefold() in schema.columns[t]
            and (not header.table or header.table.casefold()==t)]
    if len(tables)!=1:return None
    table=tables[0];keys=[n for n,c in schema.columns[table].items() if c.get('pk')]
    if len(keys)!=1:return None
    # Consume the complete supported utterance, not a substring of a richer request.
    q=question.strip().rstrip('?.').strip().casefold()
    prefixes='|'.join(re.escape(p) for p in sorted(config['grain_prefixes'],key=len,reverse=True))
    connectors='|'.join(re.escape(p) for p in config['grain_connectors'])
    m=re.fullmatch(r'(?:'+prefixes+r')\s+(?:the\s+)?(.+?)\s+(?:(?:'+connectors+r')\s+)?'+re.escape(target['evidence'])+r'\s*',q)
    if not m or words(m.group(1))!=words(header.name):return None
    # quantity() must have consumed the only numeric/entity phrase.
    return {'table':table,'group':header.name.casefold(),'pk':keys[0],'number':target['number'],
            'operator':target['operator'].__name__,'question_evidence':target['evidence'],
            'nullable_pk_unproved':not bool(schema.columns[table][keys[0]].get('not_null'))}


def name_tokens(name):
    return re.sub(r'([a-z0-9])([A-Z])',r'\1_\2',name).casefold().split('_')


def name_spec(question,original,schema,config):
    try:
        q=parse(original)
        if not isinstance(q,exp.Select) or q.args.get('with') or q.args.get('joins'):return None
        source=q.args.get('from')
        if source is None or not isinstance(source.this,exp.Table):return None
        table=source.this.name.casefold()
        if table not in schema.columns:return None
        # At least one ordinary projected column, all projections owned by this outer table.
        cols=[c for e in q.expressions for c in e.find_all(exp.Column)]
        if not cols or any(c.is_star or c.name.casefold() not in schema.columns[table] or
                           (c.table and c.table.casefold()!=source.this.alias_or_name.casefold()) for c in cols):return None
        names=[n for n,c in schema.columns[table].items() if any(x in c.get('type','').upper() for x in config['name_type_tokens'])
               and name_tokens(c['name'])[-1:] == ['name'] and not set(name_tokens(c['name'])) & set(config['name_excluded_tokens'])]
        if len(names)!=1:return None
        literals={str(n.this) for n in q.find_all(exp.Literal) if n.is_string}
        # Table words use the same existing singularization, but question keeps natural plural forms.
        variants={schema.tables[table]['name'].replace('_',' ').casefold(),' '.join(words(schema.tables[table]['name']))}
        modifiers='|'.join(re.escape(x) for x in config['name_phrase_options'])
        hits=[]
        for literal in sorted(literals):
            if any(re.search(r'\b'+re.escape(v)+r'\s+(?:(?:'+modifiers+r')\s+)?[\'\"]?'+re.escape(literal)+r'[\'\"]?(?=\s|[?.!,]|$)',question,re.I) for v in variants):hits.append(literal)
        if len(hits)!=1:return None
        return {'table':table,'column':names[0],'literal':hits[0],'question_evidence':'direct_entity_name_phrase'}
    except (ValueError,RecursionError,SqlglotError):return None


def grain_query(spec,schema):
    def col(n):return exp.column(schema.columns[spec['table']][n]['name'],table=schema.tables[spec['table']]['name'],quoted=True)
    count=exp.Count(this=exp.Distinct(expressions=[col(spec['pk'])]))
    condition=getattr(exp,spec['operator'])(this=count,expression=exp.Literal.number(spec['number']))
    return exp.select(col(spec['group'])).from_(exp.Table(this=exp.to_identifier(schema.tables[spec['table']]['name'],quoted=True))).group_by(col(spec['group'])).having(condition)


def name_variant(sql,spec,schema):
    """Replace a single positive equality/lookup; do not discard other conditions."""
    try:
        q=parse(sql)
        if not isinstance(q,exp.Select) or q.args.get('with') or q.args.get('joins'):return None
        f=q.args.get('from');w=q.args.get('where')
        if not f or not isinstance(f.this,exp.Table) or f.this.name.casefold()!=spec['table'] or not w:return None
        if any(isinstance(n,(exp.Or,exp.Not)) for n in w.walk()):return None
        sources=_sources(q)
        columns=[c for e in q.expressions for c in e.find_all(exp.Column)]
        if not columns or any(_physical_owner(c,sources,schema)!=spec['table'] for c in columns):return None
        hits=[]
        for part in _parts(w.this):
            if isinstance(part,exp.EQ) and isinstance(part.this,exp.Column) and isinstance(part.expression,exp.Literal):
                if part.expression.is_string and part.expression.this==spec['literal'] and _physical_owner(part.this,sources,schema)==spec['table']:hits.append(part)
                continue
            sub=None
            if isinstance(part,exp.EQ) and isinstance(part.this,exp.Column) and isinstance(part.expression,exp.Subquery):sub=part.expression.this
            elif isinstance(part,exp.In) and isinstance(part.this,exp.Column):
                value=part.args.get('query');sub=value.this if isinstance(value,exp.Subquery) else value
            if not isinstance(sub,exp.Select) or sub.args.get('joins') or sub.args.get('with'):continue
            if _physical_owner(part.this,sources,schema)!=spec['table'] or len(sub.expressions)!=1:continue
            if any(sub.args.get(k) for k in ('group','having','order','limit','offset')):continue
            sw=sub.args.get('where');sf=sub.args.get('from')
            if not sw or not sf or not isinstance(sf.this,exp.Table) or not isinstance(sw.this,exp.EQ):continue
            p=sw.this
            if not isinstance(p.this,exp.Column) or not isinstance(p.expression,exp.Literal) or not p.expression.is_string or p.expression.this!=spec['literal']:continue
            if any(isinstance(n,exp.Select) and n is not sub for n in sub.walk()):continue
            if len(list(sub.find_all(exp.Literal)))!=1:continue
            inner_sources=_sources(sub)
            projected=unalias(sub.expressions[0])
            if not isinstance(projected,exp.Column) or not _physical_owner(projected,inner_sources,schema):continue
            if not _physical_owner(p.this,inner_sources,schema):continue
            hits.append(part)
        if len(hits)!=1:return None
        replacement=exp.EQ(this=exp.column(schema.columns[spec['table']][spec['column']]['name'],table=f.this.alias_or_name,quoted=True),
                           expression=exp.Literal.string(spec['literal']))
        hits[0].replace(replacement)
        return q
    except (ValueError,RecursionError,SqlglotError):return None


def proposals(roots,facts):
    """Finite raw roots only; one quantity template and one name edit per root."""
    if facts.grain:
        yield roots[0] if roots else None,'entity_group_count',grain_query(facts.grain,facts.schema),facts.grain
    if facts.name:
        for parent in roots:
            q=name_variant(parent['sql'],facts.name,facts.schema)
            if q is not None:yield parent,'entity_name_binding',q,facts.name
