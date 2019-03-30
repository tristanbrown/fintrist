"""
Start Fintrist, connect to the DB, and expose public methods.
"""

import mongoengine
from . import settings
from .models import Stream, Study

__all__ = ('Stream',)

mongoengine.connect(
    settings.DATABASE_NAME,
    # username=settings.USERNAME,
    # password=settings.PASSWORD,
    # authentication_source='admin'
)

# TODO: Explicitly specify fintrist db connection for authentication.
# TODO: Implement User info in a separate db.
# TODO: Use "roles" with $redact to control access to specific documents
# TODO: https://stackoverflow.com/questions/28168258/does-mongodb-provide-document-level-access-to-users
# TODO: Alternatively, use "owner" and "group" fields on documents, and use mongoengine filters to control access
# TODO: Is this vulnerable? Example: fintrist.Stream.objects will still return all objects, regardless of auth.
# TODO: Need to deny users direct access to object operations.
# TODO: Create two db roles: "Develop" and "Flask-access" https://docs.mongodb.com/manual/core/authorization/
# TODO: https://docs.mongodb.com/manual/tutorial/manage-users-and-roles/
