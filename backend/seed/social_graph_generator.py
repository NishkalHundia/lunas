"""Generate a realistic social graph using a stochastic block model.

Why not Erdos-Renyi (uniform random)? Real social networks have:
  1. Community structure — people cluster by neighborhood + interests
  2. Triadic closure — if A knows B and B knows C, A is likely to know C
  3. Power-law degree distribution — most people have ~30 friends, some have 100+

This generator captures all three properties through:
  - Community-based edge probability (stochastic block model)
  - Mutual-friend boosting pass (triadic closure)
  - Cross-cluster random edges (weak ties / power-law tail)
"""

import random
import uuid
from datetime import datetime, timezone

from backend.models.user import User
from backend.models.friendship import Friendship

INTRA_COMMUNITY_EDGE_PROB = 0.12
INTER_COMMUNITY_EDGE_PROB = 0.008
MUTUAL_FRIEND_BOOST_PROB = 0.15
CROSS_CLUSTER_RANDOM_EDGES = 500


def _community_key(user: User) -> str:
    return f"{user.neighborhood or 'Unknown'}|{user.archetype or 'casual'}"


def generate_social_graph(users: list[User], rng_seed: int = 42) -> list[Friendship]:
    random.seed(rng_seed)

    communities: dict[str, list[str]] = {}
    user_map: dict[str, User] = {}
    for u in users:
        key = _community_key(u)
        communities.setdefault(key, []).append(u.id)
        user_map[u.id] = u

    user_ids = [u.id for u in users]
    adjacency: dict[str, set[str]] = {uid: set() for uid in user_ids}
    edges: set[tuple[str, str]] = set()

    def _add_edge(a: str, b: str):
        if a == b:
            return
        key = (min(a, b), max(a, b))
        if key not in edges:
            edges.add(key)
            adjacency[a].add(b)
            adjacency[b].add(a)

    for comm_key, members in communities.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if random.random() < INTRA_COMMUNITY_EDGE_PROB:
                    _add_edge(members[i], members[j])

    comm_keys = list(communities.keys())
    for ci in range(len(comm_keys)):
        for cj in range(ci + 1, len(comm_keys)):
            m1 = communities[comm_keys[ci]]
            m2 = communities[comm_keys[cj]]
            for u1 in m1:
                for u2 in m2:
                    if random.random() < INTER_COMMUNITY_EDGE_PROB:
                        _add_edge(u1, u2)

    all_pairs_to_check = []
    for uid in user_ids:
        friends = adjacency[uid]
        for fid in friends:
            for fof in adjacency[fid]:
                if fof != uid and fof not in adjacency[uid]:
                    all_pairs_to_check.append((uid, fof))

    random.shuffle(all_pairs_to_check)
    for a, b in all_pairs_to_check[:len(all_pairs_to_check) // 3]:
        mutual_count = len(adjacency[a] & adjacency[b])
        if mutual_count >= 2 and random.random() < MUTUAL_FRIEND_BOOST_PROB:
            _add_edge(a, b)

    for _ in range(CROSS_CLUSTER_RANDOM_EDGES):
        a, b = random.sample(user_ids, 2)
        _add_edge(a, b)

    friendships: list[Friendship] = []
    for (a, b) in edges:
        score = random.uniform(0.0, 1.0) * (0.3 if _community_key(user_map[a]) == _community_key(user_map[b]) else 0.1)
        friendships.append(Friendship(
            id=str(uuid.uuid4()),
            user_id=a,
            friend_id=b,
            status="accepted",
            relationship_type="friend",
            interaction_score=round(score, 3),
            created_at=datetime.now(timezone.utc),
        ))

    return friendships
