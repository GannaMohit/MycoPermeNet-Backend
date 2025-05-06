import graphene

from api.schema.atomic import AtomicQuery

class Query(
    AtomicQuery,
    graphene.ObjectType
):
    pass


schema = graphene.Schema(query=Query)