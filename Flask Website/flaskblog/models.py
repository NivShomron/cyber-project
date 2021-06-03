from flaskblog import db, login_manager
from flask_login import UserMixin


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    features = db.relationship('Features', backref='user')

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"


class Features(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('user.id'), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    country = db.Column(db.Text, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Text, nullable=False)
    result = db.Column(db.Text, nullable=False)
    tested = db.Column(db.Text, nullable=True)

    def set_tested(self, tstd):
        self.tested = tstd

    def __repr__(self):
        value = f"Age: {self.age}, Gender: {self.gender}, Country: {self.country} \n Symptoms: {self.symptoms} \n Result: {self.result} \n Tested result: {self.tested}"
        return value
