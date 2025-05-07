import graphene

from api.schema.atomic import AtomicQuery
from api.schema.molecular import MolecularQuery

class Query(
    AtomicQuery,
    MolecularQuery,
    graphene.ObjectType
):
    pass


schema = graphene.Schema(query=Query)