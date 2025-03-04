"""
Social dynamics package for relationships, networks, and community formation.
"""

from social.network import SocialNetwork
from social.relationship import Relationship, RelationshipPool
from social.community import Community, CommunityDetector

try:
    from social.interactions import calculate_relationship_strength, evaluate_social_interactions
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

__all__ = [
    'SocialNetwork',
    'Relationship',
    'RelationshipPool',
    'Community',
    'CommunityDetector',
    'HAS_CYTHON'
]

if HAS_CYTHON:
    __all__ += ['calculate_relationship_strength', 'evaluate_social_interactions']