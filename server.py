from flask import Flask, request, redirect, flash, render_template, send_from_directory, jsonify
import os as o
from flask.helpers import url_for
from stitcher import start
from werkzeug.utils import secure_filename
import glob as gb

server = Flask(__name__, template_folder='static')
server.secret_key = 'secret_key'

# util functions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = o.path.join(o.getcwd(), 'images')


def checkFileType(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


@server.route('/', methods=['GET', 'POST'])
def home():
    # print(request.files)
    file_names = None
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files')
        print(files)
        file_names = []
        for file in files:
            print(file)
            if file and checkFileType(file.filename):
                print(True)
                filename = secure_filename(file.filename)
                file_names.append(filename)
                print(filename)
                file.save(o.path.join(UPLOAD_FOLDER, filename))

                return redirect(request.url)

    if request.method == 'GET':
        image_names = o.listdir('./images')
        print(image_names)
        return render_template("home.html", image_names=image_names)

    return render_template('home.html', filenames=file_names)

@server.route('/display/<filename>')
def sendImage(filename):
    if filename == 'results':
        return send_from_directory("results", 'final.png')
    return send_from_directory("images", filename)

@server.route('/process')
def displayImages():
    image_names = o.listdir('./images')
    print(image_names)
    return render_template("process.html", image_names=image_names)

@server.route('/start', methods=['GET'])
def begin():
    try:
        process = start()
    except Exception as e:
        flash('error - ' + str(e) + ' - clear cache and retry')
    flash('process - completed')
    return redirect('/')

@server.route('/clear', methods=['GET'])
def cacheClear():
    image_path = o.path.join(o.getcwd(), "images")
    result_path = o.path.join(o.getcwd(), "results")

    files = o.listdir(image_path)
    for f in files:
        o.remove(o.path.join(image_path, f))
    
    files = o.listdir(result_path)
    for f in files:
        o.remove(o.path.join(result_path, f))

    return redirect('/')

if __name__ == '__main__':
    server.run(host='0.0.0.0',debug=True, port=3000)
