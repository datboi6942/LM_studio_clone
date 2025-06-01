"""Admin interface routes."""

from flask import Blueprint

bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.route("/")
def index():
    """Admin dashboard."""
    return "Admin UI coming soon!"  # TODO: Implement admin panel 