import graphene

from api.schema.atomic import AtomicQuery
from api.schema.molecular import MolecularQuery
from api.schema.similarity import SimilarityQuery

class Query(
    AtomicQuery,
    MolecularQuery,
    SimilarityQuery,
    graphene.ObjectType
):
    pass


schema = graphene.Schema(query=Query)