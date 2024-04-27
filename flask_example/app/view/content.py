from flask import (
    Blueprint, flash, redirect, render_template, request, url_for, current_app
)
from flask import g

from app.db import get_db

from app.utils import login_required

bp = Blueprint('content', __name__, url_prefix='/content')


@bp.route('/edit', methods=('GET', 'POST'))
@login_required
def edit():
    contents = get_db().execute(
        'SELECT contents FROM user WHERE id = ?', (g.user['id'],)
    ).fetchone()['contents']
    if request.method == "POST":
        contents = request.form["contents"]
        error = None

        if not contents:
            error = "content is required."

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                "UPDATE user SET contents = ? WHERE id = ?", (contents, g.user['id'])
            )
            current_app.config['rag'].save_content(user=str(g.user['id']), content=contents, content_type="user_contents")
            db.commit()
            return redirect(url_for("content.edit"))

    return render_template("content/edit.html", contents=contents)
