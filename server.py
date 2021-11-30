from flask import Flask, request as req, redirect, flash, render_template, send_from_directory, jsonify
import os as o
from flask.helpers import url_for
from stitcher import pStart
from werkzeug.utils import secure_filename

server = Flask(__name__, template_folder='static')
server.secret_key = 'secret_key'

# util functions
ALLWD_EXTNS = set(['png', 'jpg', 'jpeg'])
UP_FLDR = o.path.join(o.getcwd(), 'images')


def checkFileType(nameF):
    return '.' in nameF and nameF.rsplit(
        '.', 1)[1].lower() in ALLWD_EXTNS


@server.route('/', methods=['GET', 'POST'])
def home():
    # print(request.files)
    file_names = None
    if req.method == 'POST':
        if 'files' not in req.files:
            flash('multi part file not found')
            return redirect(req.url)
        files = req.files.getlist('files')
        print(files)
        file_names = []
        for xf in files:
            print(xf)
            if xf and checkFileType(xf.filename):
                print(True)
                filename = secure_filename(xf.filename)
                file_names.append(filename)
                print(filename)
                xf.save(o.path.join(UP_FLDR, filename))

                return redirect(req.url)

    if req.method == 'GET':
        image_names = o.listdir('./images')
        print(image_names)
        return render_template("home.html", image_names=image_names)

    return render_template('home.html', filenames=file_names)

@server.route('/display/<filename>')
def sendImage(filename):
    if filename == 'results':
        return send_from_directory("results", 'stitfinal.png')
    return send_from_directory("images", filename)

@server.route('/start', methods=['GET'])
def begin():
    try:
        pStart()
    except Exception as xerr:
        flash('error - ' + str(xerr) + ' - clear cache and retry')
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
  
