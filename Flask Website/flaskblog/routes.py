from flask import render_template, url_for, flash, redirect, request
from flaskblog import app, db, bcrypt
from flaskblog.forms import RegistrationForm, LoginForm
from flaskblog.models import User, Features
from flask_login import login_user, current_user, logout_user, login_required
from flaskblog.ml_model import model_pred, reshape_arr


@app.route("/")
@app.route("/home")
def home():
    """
    If opens the website (navigates to /) or navigates to /home, home.html will be loaded
    """
    return render_template('home.html')


@app.route("/about")
def about():
    """
    If user navigates to /about, about.html will be loaded
    """
    return render_template('about.html')


@app.route("/register", methods=['GET', 'POST'])
def register():
    """
    First checks if user is already logged in, if he is he will be sent to the home page.
    Checks if the forms are correct, if so, will try to add the use to the database and will give an alert
    message accordingly. if there is an error, will return to the register page and will show user what went wrong.
    """
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        try:
            user = User(username=form.username.data, email=form.email.data, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash('Your account has been created! You are now able to log in', 'success')
            return redirect(url_for('login'))
        except:
            flash('There is already an account linked to this username or email, please try again with a different one',
                  'danger')
            return redirect(url_for('register'))

    return render_template('register.html', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    """
    First checks if user is already logged in, if he is he will be sent to the home page.
    Checks if the forms are correct by filtering the database to find only the given email. The given password
    will be hashed, in order to compare it to the hash password saved in the database. If the given email or
    password wasn't correct, an alert will be shown.
    """
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)


@app.route("/logout")
def logout():
    """
    If user navigates to /logout, user will be logged out, and he will be redirected to the home function
    """
    logout_user()
    return redirect(url_for('home'))


@app.route("/account")
@login_required
def account():
    """
    If user navigates to /account, will check if user is logged in. If he is, will receive all info about the
    user, pass it to the account.html file, in order for it to be shown on his account page.
    """
    if current_user.is_authenticated:
        feat = Features.query.filter_by(user_id=current_user.username)
        try:
            age = feat.order_by(Features.id.desc()).first().age

            gender = feat.order_by(Features.id.desc()).first().gender
            if gender == "1":
                gender = "Male"
            else:
                gender = "Female"

            country = feat.order_by(Features.id.desc()).first().country
            symptoms = feat.order_by(Features.id.desc()).first().symptoms

            result = feat.order_by(Features.id.desc()).first().result
            if result == "1":
                result = "Positive"
            else:
                result = "Negative"

            # data=1 is passed if the user has any data to be shown on profile - got evaluated
            tested = feat.order_by(Features.id.desc()).first().tested
            print(tested)
            return render_template('account.html', data="1", age=age, gndr=gender, cntry=country,
                                   symptoms=symptoms, rslt=result, tstd=tested)
        except:
            return render_template('account.html', data="0")


@app.route("/eval", methods=['GET', 'POST'])
def eval():
    """
    If user navigates to /eval, will check if method is POST. If it is, will receive all info which was put
    into the form. If user didn't select country (if it is still on the "Select Country" option),
    will load eval.html again. Otherwise, the data will be put in the array which will be reshaped into
     a numpy array and put into the model in order to be predicted. When the model returns a prediction,
      it will be added to the database and load result.html with the result.
    """
    if request.method == "POST":
        age = request.form["age"]
        gender = request.form["gender"]
        country = request.form["country"]
        cough = request.form["cough"]
        fever = request.form["fever"]
        sorethroat = request.form["sorethroat"]
        shortness = request.form["shortness"]
        headache = request.form["headache"]
        contact = request.form["contact"]

        # if didn't select a valid country
        if country == "select":
            flash('Please select a valid country', 'danger')
            return render_template('eval.html', title='Evaluation')

        arr = [cough, fever, sorethroat, shortness, headache, contact, str(int(int(age) / 60)), gender]

        numpy_data = reshape_arr(arr)

        pred = model_pred(numpy_data)
        # add features to the database - if not logged in, user_id will be guest
        if current_user.is_authenticated:
            features = Features(user_id=current_user.username,
                                symptoms=f"{cough}{fever}{sorethroat}{shortness}{headache}{contact}", country=country,
                                age=age, gender=gender, result=str(pred[0]))
        else:
            features = Features(user_id="guest", symptoms=f"{cough}{fever}{sorethroat}{shortness}{headache}{contact}",
                                country=country, age=age, gender=gender, result=str(pred[0]))
        db.session.add(features)
        db.session.commit()

        if pred[0] == 0:
            return render_template("result.html", rslt="Negative", age=age, gndr=gender, cntry=country, cough=cough,
                                   fever=fever, srtrt=sorethroat, shrtnss=shortness, hdche=headache, cntct=contact)
        elif pred[0] == 1:
            return render_template("result.html", rslt="Positive", age=age, gndr=gender, cntry=country, cough=cough,
                                   fever=fever, srtrt=sorethroat, shrtnss=shortness, hdche=headache, cntct=contact)

    else:
        return render_template('eval.html', title='Evaluation')


@app.route("/result")
def result():
    """
    If user navigates to /result, result.html will be loaded
    """
    return render_template('result.html', title='Evaluation Result')


@app.route("/tested", methods=['GET', 'POST'])
def tested():
    """
    If method is POST, will try to receive the input from the form and insert the value into the current
    user's database row. Will send an alert regarding the result of the action, and if doesn't work, will
    redirect back to the eval function. If method isn't POST, will load tested.html
    """
    if request.method == "POST":
        try:
            tested = request.form["tested"]
            feat = Features.query.filter_by(user_id=current_user.username)

            feat.order_by(Features.id.desc()).first().tested = tested
            db.session.commit()

            flash('Your test result has been submitted. Thanks for helping us improve our model!', 'success')
            return render_template('home.html')
        except:
            flash('In order to input your test result you first need to get evaluated', 'danger')
            return redirect(url_for('eval'))

    return render_template('tested.html', title='Inputting Tested Result')


@app.route("/graphs")
def graphs():
    """
    If user navigates to /graphs, this function will load graphs.html file
    """
    return render_template('graphs.html')


if __name__ == '__main__':
    app.run(debug=True)
