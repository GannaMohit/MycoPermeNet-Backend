import graphene

from atomic.schema import AtomicQuery

class Query(
    AtomicQuery,
    graphene.ObjectType
):
    pass


schema = graphene.Schema(query=Query)