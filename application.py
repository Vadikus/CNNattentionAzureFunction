from flask import Flask
myapp = Flask(__name__)

@myapp.route("/")
def app():
    return "Hello Flask, on Azure App Service for Linux"