"""adds reconstruction to without fitting semantic weights or new exits."""
from collections import deque
import time

from mp3_sql.sqlite_tool import validate_sql
from mp3_agent.safety import sql_key
from mp3_agent.semantic_agent import SemanticAgent
from mp3_agent.skeleton_rebuild import ROUTES, proposals


def ordinary_origins(row):
    return [o for o in row['origins'] if not o.get('skeleton_route')]


class SkeletonAgent(SemanticAgent):
    def __init__(self, config, log=None):
        super().__init__(config, log)
        self.skeleton_done = False
        self.skeleton_events = []
        self.skeleton_keys = set()
        self.skeleton_seconds = 0.
        self.skeleton_anchors = []

    def baseline_only(self, operation):
        """Keep the approved one-level reconstruction out of both old edit closures."""
        all_rows = self.candidates
        self.candidates = {k: {**r, 'origins': ordinary_origins(r)} for k, r in all_rows.items() if ordinary_origins(r)}
        try:
            return operation()
        finally:
            for key, row in self.candidates.items():
                if key not in all_rows:
                    all_rows[key] = row
                else:
                    for origin in row['origins']:
                        if origin not in all_rows[key]['origins']:
                            all_rows[key]['origins'].append(origin)
            self.candidates = all_rows

    def expand(self):
        if not self.config.get('skeleton_enabled'):
            return super().expand()
        return self.baseline_only(super().expand)

    def diagnostic_rows(self):
        # No new probe/edit descendants of a pure reconstruction; merged old origins retain eligibility.
        return (r for r in self.candidates.values() if ordinary_origins(r)) if self.config.get('skeleton_enabled') else super().diagnostic_rows()

    def generate(self):
        if not self.config.get('skeleton_enabled'):
            return super().generate()
        self.baseline_only(super().generate)
        if self.skeleton_done:
            return
        self.skeleton_done = True
        started = time.perf_counter()
        roots = []
        for row in self.candidates.values():
            raw = [o for o in row['origins'] if o['kind'] in ('original', 'hint')]
            if raw:
                roots.append((min(o['root_order'] for o in raw), row))
        roots.sort(key=lambda x: (x[0], x[1]['key']))
        queue = deque()
        for order, row in roots:
            self.skeleton_anchors.append({'key': row['key'], 'root_order': order})
            for route in ROUTES:
                events = []
                self.skeleton_events.append({'anchor': row['key'], 'route': route, 'events': events})
                gen = proposals(row['sql'], self.constraints.labels, self.schema, self.facts.question,
                                self.facts.count_table, self.config, route, events)
                queue.append((order, row, route, gen, events))
        try:
            while queue:
                if len(self.skeleton_keys) >= self.config['max_skeleton_candidates']:
                    self.skeleton_events.append({'reason': 'skeleton_candidate_cap', 'pending_groups': len(queue)})
                    break
                order, parent, route, gen, events = queue.popleft()
                # A group yields one distinct added SQL per fair round; duplicates retain lineage.
                for item in gen:
                    events.append({'reason': 'proposal', 'sql': item['sql']})
                    try:
                        key = sql_key(item['sql'])
                    except (ValueError, RecursionError) as e:
                        self.rejections.append({'sql': item['sql'], 'reason': str(e), 'skeleton_route': route})
                        continue
                    fresh = key not in self.candidates
                    origin = {'kind': 'rewrite', 'parent_key': parent['key'], 'root_sql': parent['sql'],
                              'root_order': order, 'steps': 1, 'tie': [2, order, 6, ROUTES.index(route)],
                              'skeleton_route': route, 'skeleton_depth': 1,
                              'path': [{'rule': 'skeleton', 'evidence': item, 'risks': item['risks']}]}
                    state, bad = self.facts.schema_state(self.tree_for_sql(item['sql']))
                    if state is False:
                        self.rejections.append({'sql': item['sql'], 'origin': origin, 'reason': 'skeleton_known_schema_error', 'columns': bad})
                        continue
                    row = self.add(item['sql'], origin)
                    if row and fresh:
                        self.skeleton_keys.add(key)
                        queue.append((order, parent, route, gen, events))
                        break
                # Exhausted groups leave the queue. Cap is shared across P/S/J and all anchors.
        finally:
            self.skeleton_seconds += time.perf_counter()-started

    @staticmethod
    def tree_for_sql(sql):
        from mp3_agent.semantic_ops import normalize, parse
        return normalize(parse(sql))

    def finish(self, observation, reason):
        log = self.log
        self.log = None
        try:
            action = super().finish(observation, reason)
        finally:
            self.log = log
        sql_key(action['sql'])
        validate_sql(action['sql'])
        self.record['skeleton'] = {'enabled': self.config['skeleton_enabled'], 'anchors': self.skeleton_anchors,
                                   'new_keys': sorted(self.skeleton_keys), 'events': self.skeleton_events,
                                   'generation_seconds': self.skeleton_seconds, 'max_depth': 1 if self.skeleton_keys else 0}
        self.record['configuration'].update(batch_id=self.config['batch_id'], skeleton_enabled=self.config['skeleton_enabled'])
        if log:
            log.finish_task(self.record)
        return action
