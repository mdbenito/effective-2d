#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import OrderedDict
import json
import io
import re
import shutil
from plots import *
import matplotlib.pyplot as pl
import pickle as pk
from math import floor, log10
import binascii
import os


class Data(object):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def columns(self):
        pass


class PickleData(Data):
    def __init__(self, model):
        super().__init__()
        self.results_file = 'results-%s-combined.pickle' % model
        try:
            print("Loading database... ", end='')
            with open(self.results_file, "rb") as fd:
                results = pk.load(fd)
            print("done.")
        except FileNotFoundError:
            print("ERROR: results file '%s' not found" % self.results_file)
            results = {}
        self._results = OrderedDict(sorted(results.items(), key=lambda x: x[1]['theta']))

    def __getitem__(self, item):
        return self._results.get(item)

    def delete(self, item):
        # TODO: save modified results to disk...
        try:
            # from sys import stderr
            # stderr.write("Would delete: " + item)
            del self._results[item]
            return item
        except KeyError:
            return None

    def columns(self):
        cols = [{'name': 'form_name', 'title': 'Q2'},
                {'name': 'init', 'title': 'Initial data'},
                {'name': 'theta', 'title': 'theta', 'type': 'number'},
                {'name': 'mu', 'title': 'mu', 'type': 'number'},
                {'name': 'e_stop', 'title': 'eps order', 'type': 'number'},
                {'name': 'steps', 'title': 'Steps', 'type': 'number'},
                {'name': 'time', 'title': 'Duration', 'type': 'time'},
                {'name': 'plot', 'title': '', 'type': 'html'},
                {'name': 'results', 'type': 'html', 'breakpoints': 'all', 'title': 'results file:'},
                {'name': 'form_arguments', 'breakpoints': 'all', 'title': 'form arguments:'},
                {'name': 'select', 'title': ''}]
        return json.dumps(cols)

    def rows(self):
        def toggle_button(id:str) -> str:
            input = '<input class="tgl tgl-flat" id="%s" type="checkbox"/>' % id
            label = '<label class="tgl-btn" for="%s"></label>' % id
            return input+label

        ret = []
        for key, row in self._results.items():
            row_data = {col: val for col, val in row.items() if col in ('init', 'steps')}
            row_data['theta'] = round(row['theta'], 3)
            row_data['mu'] = round(row['mu'], 2)
            tt = max(0, row.get('time', 0))
            hh = floor(tt / 3600)
            mm = floor((tt - hh * 3600) / 60)
            ss = floor((tt - hh * 3600 - mm * 60))
            row_data['time'] = "%02d:%02d:%02d" % (hh, mm, ss)
            row_data['form_name'] = row.get('Q2', {}).get('form_name', 'N/A')
            row_data['e_stop'] = int(log10(row.get('e_stop', 1))) - 1
            row_data['plot'] = '<a class="plot_one" href="#single_plot_target" data-value="%s" onclick="show_one(this)">Plot</a>' % key
            file_name = os.path.join(os.getcwd(), row.get('file_name', 'UNAVAILABLE'))
            row_data['results'] = '<a href="pvd://%s">%s</a>' % (file_name, os.path.basename(file_name))
            row_data['form_arguments'] = ", ".join("%s: %f" % (k, v) for k, v in
                                                    row.get('Q2', {}).get('arguments', {}).items())
            row_data['select'] = toggle_button(key)
            ret.append(row_data)
        return json.dumps(ret)

    def get_multiple(self, ids:list) -> dict:
        return {k:v for k, v in self._results.items() if k in ids}


class Handler(BaseHTTPRequestHandler):
    """
    GET API:
      /columns
      /rows
      /plot_one/[key]
    POST API:
      /plot_multiple [list of keys]
    """

    def _set_headers(self, *args, content='application/json', charset='utf8'):
        self.send_response(*args)
        self.send_header('Content-type', content)
        self.send_header('charset', charset)
        self.end_headers()

    def _image_headers(self):
        self.send_response(200)
        self.send_header('Pragma', 'public')
        self.send_header('Cache-Control', 'max-age=86400')
        # self.send_header('Expires', gmdate('D, d M Y H:i:s \G\M\T', time() + 86400));
        self.send_header('Content-Type', 'image/png')
        self.end_headers()

    def do_GET(self):
        if self.path in ("", "/"):
            self.path = "/index.html"
        if re.match('/(css/.*|js/.*|fonts/.*|index.*)', self.path) is not None:
            mimetypes = {
                "html": "text/html",
                "js": "text/javascript",  # I guess these two are wrong but who cares
                "css": "text/css",
                "eot": "application/vnd.ms",
                "otf": "application/font-sfnt",
                "svg": "image/svg+xml",
                "ttf": "application/font-sfnt",
                "woff": "application/font-woff",
                "woff2": "font/woff2"}
            fname = self.path.split('?')[0]  # ignore url arguments
            with open("report/" + fname.lstrip('/'), "rb") as fd:
                #self.log_message("Serving file: %s" % fname)
                extension = self.path[self.path.rfind('.')+1:]
                self._set_headers(200, content=mimetypes.get(extension, "text/plain"))
                shutil.copyfileobj(fd, self.wfile)

        elif self.path.endswith('/api/reload'):
            global data
            data = PickleData('curl')
            self._set_headers(200)

        elif self.path.endswith('/api/columns'):
            self._set_headers(200)
            # with open("report/columns.json", "rb") as fd:
            #     shutil.copyfileobj(fd, self.wfile)
            self.wfile.write(bytes(data.columns(), 'utf-8'))

        elif self.path.endswith('/api/rows'):
            self._set_headers(200)
            # with open("report/rows.json", "rb") as fd:
            #     shutil.copyfileobj(fd, self.wfile)
            self.wfile.write(bytes(data.rows(), 'utf-8'))

        elif re.match('/api/plot_one/', self.path) is not None:
            key = self.path.split('/')[-1]
            if data[key] is not None:
                if data[key]['steps'] < 200:
                    beg, win = 10, 1
                elif data[key]['steps'] < 1000:
                    beg, win = 40, 20
                else:
                    beg, win = 50, 50
                with io.BytesIO() as buf:
                    try:
                        plots1(data[key], slice(beg, -1), win)
                        pl.savefig(buf, format='png')
                        pl.close()
                        buf.seek(0)
                        self._image_headers()
                        shutil.copyfileobj(buf, self.wfile)
                    except Exception as e:
                        self._set_headers(400, 'Error: %s' % str(e))
            else:
                self.send_error(400, 'Bad Request: run "%s" does not exist' % key)
        elif re.match('/api/plot_multiple/', self.path) is not None:
            ids = self.path.split('/')[-1].split(',')
            steps = min([data[id]['steps'] for id in ids])
            if steps < 200:
                beg, win = 10, 1
            elif steps < 1000:
                beg, win = 40, 20
            else:
                beg, win = 50, 50
            with io.BytesIO() as buf:
                try:
                    plots4(data.get_multiple(ids), slice(beg, -1), win)
                    pl.savefig(buf, format='png')
                    pl.close()
                    buf.seek(0)
                    self._image_headers()
                    b64 = binascii.b2a_base64(buf.getvalue())
                    shutil.copyfileobj(io.BytesIO(b64), self.wfile)
                except Exception as e:
                    self.send_error(400, "Error: %s" % str(e))
        elif re.match('/api/delete/', self.path) is not None:
            ids = self.path.split('/')[-1].split(',')
            deleted = []
            for id in ids:
                deleted.append(data.delete(id))
            self._set_headers(200)
            json.dumps(deleted)
        else:
            # self.log_error("Unknown API call: '%s'" % self.path)
            self.send_error(403, "Unknown API call: '%s'" % self.path)

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        #self.log_error("Unhandled POST API call: '%s'" % self.path)
        self.send_error(403, "Unhandled POST API call: '%s'" % self.path)


def run(server_class=HTTPServer, handler_class=Handler, port=8080):
    server_address = ('localhost', port)
    httpd = server_class(server_address, handler_class)
    print('Starting server...')
    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv

    pl.ioff()
    # HACK: remove with DB connection
    data = PickleData('curl')

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
