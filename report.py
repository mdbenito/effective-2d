#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import OrderedDict
import json
import io
import re
import shutil
import importlib
import plots
import matplotlib.pyplot as pl
import pickle as pk
from math import floor, log10
import binascii
import os


class PickleData(object):
    """ A simple interface around the dict of results with all runs.

    Data is stored in a pickle, so it is necessarily written only in
    bulk with save(). This is slow and can cause data loss if the server
    writes at the same time that some simulation does, for instance.
    TODO: ditch this and use some database.
    TODO: templatize the format of row data (css classes etc.)
    """
    def __init__(self, filename):
        super().__init__()
        self.results_file = filename
        self._results = {}
        self.load()

    def load(self):
        try:
            print("Loading database... ", end='')
            with open(self.results_file, "rb") as fd:
                results = pk.load(fd)
            print("done.")
            self._results = OrderedDict(sorted(results.items(), key=lambda x: x[1]['theta']))
        except FileNotFoundError:
            print("ERROR: results file '%s' not found" % self.results_file)

    def __getitem__(self, item_s):
        if isinstance(item_s, str):
            return self._results.get(item_s, {})
        elif isinstance(item_s, list):
            return {k: v for k, v in self._results.items() if k in item_s}
        else:
            return None

    def save(self):
        """ Dump all of the data back into the pickle.
        TODO: don't convert back to dict upon saving?
        """
        try:
            print("Saving database... ", end='')
            with open(self.results_file, "wb") as fd:
                pk.dump(dict(self._results), fd)
            print("done.")
        except Exception as e:  # Pokemon!
            print("ERROR: " + str(e))

    def delete(self, item):
        """ Deletes one item from the store.
        Remember that the data is not saved to disk unless save() is called.
        """
        try:
            # from sys import stderr
            # stderr.write("Would delete: " + item)
            del self._results[item]
            return item
        except KeyError:
            return None

    def columns(self):
        """ Returns a json string with the colum data for footable. """
        cols = [{'name': 'impl', 'title': 'Impl.'},
                {'name': 'form_name', 'title': 'Q2'},
                {'name': 'init', 'title': 'Initial data'},
                {'name': 'theta', 'title': 'theta', 'type': 'number'},
                {'name': 'mu', 'title': 'mu', 'type': 'number'},
                {'name': 'e_stop', 'title': 'eps order', 'type': 'number'},
                {'name': 'steps', 'title': 'Steps', 'type': 'number', 'filterable': False},
                {'name': 'time', 'title': 'Duration', 'type': 'time', 'filterable': False},
                {'name': 'plot', 'title': '', 'type': 'html', 'filterable': False},
                {'name': 'results', 'type': 'html', 'breakpoints': 'all',
                 'title': 'results file:', 'filterable': False},
                {'name': 'form_arguments', 'breakpoints': 'all',
                 'title': 'form arguments:', 'filterable': False},
                {'name': 'mesh', 'title': 'Mesh file', 'breakpoints': 'all'},
                {'name': 'select', 'title': '', 'filterable': False}]
        return json.dumps(cols)

    def rows(self):
        """ Returns a json dict with the row data for footable. """
        def toggle_button(id:str) -> str:
            input = '<input class="tgl tgl-flat" id="%s" type="checkbox"/>' % id
            label = '<label class="tgl-btn" for="%s"></label>' % id
            return input+label

        ret = []
        for key, row in self._results.items():
            row_data = {col: val for col, val in row.items()
                        if col in ('impl', 'init', 'steps')}
            row_data['theta'] = round(row['theta'], 3)
            row_data['mu'] = round(row['mu'], 2)
            # HACK: *sometimes* data are stored in numpy formats, which json.dumps()
            # cannot handle. With item() we convert them to native python types.
            try:
                row_data['theta'] = row_data['theta'].item()
                row_data['mu'] = row_data['mu'].item()
            except AttributeError:
                pass
            tt = max(0, row.get('time', 0))
            hh = floor(tt / 3600)
            mm = floor((tt - hh * 3600) / 60)
            ss = floor((tt - hh * 3600 - mm * 60))
            row_data['time'] = "%02d:%02d:%02d" % (hh, mm, ss)
            row_data['form_name'] = row.get('Q2', {}).get('form_name', 'N/A')
            row_data['e_stop'] = int(log10(row.get('e_stop', 1))) - 1
            row_data['plot'] = '<a class="plot_one" href="#single_plot_target" ' \
                               + 'data-value="%s" onclick="show_one(this)">Plot</a>' % key
            mesh_file = row.get('mesh', '')
            row_data['mesh'] = '<a class="plot_one" href="#single_plot_target" ' \
                               + 'data-value="%s" onclick="show_mesh(this)">%s</a>' % (mesh_file, mesh_file)
            file_name = os.path.join(os.getcwd(), row.get('file_name', 'UNAVAILABLE'))
            row_data['results'] = '<a href="pvd://%s">%s</a>' % (file_name, os.path.basename(file_name))
            row_data['form_arguments'] = ", ".join("%s: %f" % (k, v) for k, v in
                                                    row.get('Q2', {}).get('arguments', {}).items())
            row_data['select'] = toggle_button(key)
            ret.append(row_data)

        return json.dumps(ret)


class Handler(BaseHTTPRequestHandler):
    """
    GET API:
      /columns
      /rows
      /plot_one/[key]
      /plot_multiple/key[,key]*
      /reload
      /delete/key[,key]*

    ****************************************************************************
    CAREFUL: delete is NOT SAFE!! There are a couple of obvious checks but they
    can probably be circumvented
    ****************************************************************************
    """

    def _set_headers(self, *args, content='application/json', charset='utf8'):
        self.send_response(*args)
        self.send_header('Content-type', content)
        self.send_header('charset', charset)
        self.end_headers()

    def _image_headers(self):
        self.send_response(200)
        self.send_header('Pragma', 'public')
        self.send_header("Pragma-directive", "no-cache")
        self.send_header("Cache-directive", "no-cache")
        self.send_header("Cache-control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        # self.send_header('Expires', gmdate('D, d M Y H:i:s \G\M\T', time() + 86400));
        self.send_header('Content-Type', 'image/png')
        self.end_headers()

    def do_GET(self):
        """ Answers GET requests. """

        # Emtpy paths lead to /index.html
        if self.path in ("", "/"):
            self.path = "/index.html"
        if re.match('/(css/.*|js/.*|fonts/.*|index.*|favicon.*.ico)', self.path) is not None:
            mimetypes = {
                "html": "text/html",
                "js": "text/javascript",  # I guess these two are wrong but who cares
                "css": "text/css",
                "eot": "application/vnd.ms",
                "otf": "application/font-sfnt",
                "svg": "image/svg+xml",
                "ttf": "application/font-sfnt",
                "woff": "application/font-woff",
                "woff2": "font/woff2",
                "ico": "image/x-icon"}
            fname = self.path.split('?')[0]  # ignore url arguments
            with open("report/" + fname.lstrip('/'), "rb") as fd:
                #self.log_message("Serving file: %s" % fname)
                extension = self.path[self.path.rfind('.')+1:]
                self._set_headers(200, content=mimetypes.get(extension, "text/plain"))
                shutil.copyfileobj(fd, self.wfile)

        elif self.path.endswith('/api/reload'):
            data.load()
            # HACK in order to tweak the plots on the fly
            importlib.reload(plots)
            self._set_headers(200)
            # quite redundant, but jQuery expects something
            self.wfile.write(bytes(json.dumps({'command':'reload', 'status': 'ok'}), 'utf-8'))

        elif self.path.endswith('/api/columns'):
            self._set_headers(200)
            self.wfile.write(bytes(data.columns(), 'utf-8'))

        elif self.path.endswith('/api/rows'):
            self._set_headers(200)
            self.wfile.write(bytes(data.rows(), 'utf-8'))

        elif re.match('/api/plot_one/', self.path) is not None:
            key = self.path.split('/')[-1]
            if data[key] is not None:
                with io.BytesIO() as buf:
                    try:
                        plots.plots1(data[key], None, None)
                        pl.savefig(buf, format='png')
                        pl.close()
                        buf.seek(0)
                        self._image_headers()
                        shutil.copyfileobj(buf, self.wfile)
                    except Exception as e:
                        self._set_headers(400, 'Error: %s' % str(e))
            else:
                self.send_error(400, 'Bad Request: run "%s" does not exist' % key)

        elif re.match('/api/plot_mesh/', self.path) is not None:
            mesh_file = self.path.split('/')[-1]
            with io.BytesIO() as buf:
                try:
                    plots.plot_mesh(mesh_file)
                    pl.savefig(buf, format='png')
                    pl.close()
                    buf.seek(0)
                    self._image_headers()
                    shutil.copyfileobj(buf, self.wfile)
                except Exception as e:
                    self._set_headers(400, 'Error: %s' % str(e))

        elif re.match('/api/plot_multiple/', self.path) is not None:
            ids = self.path.split('/')[-1].split(',')

            with io.BytesIO() as buf:
                try:
                    plots.plots4(data[ids], None, None)
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
            files_deleted = 0
            for id in ids:
                dir, file = os.path.split(data[id].get('file_name', ''))
                try:
                    # Be extra careful: a "safe" path to delete is not absolute, not .,
                    # and contains only vtu and pvd files
                    if not os.path.isabs(dir) and dir != "" and dir[0] != ".":
                        dir = os.path.join(os.getcwd(), dir)
                        for f in os.listdir(dir):
                            _, ext = os.path.splitext(f)
                            if ext in ('.pvd', '.vtu'):
                                os.unlink(os.path.join(dir, f))
                                files_deleted += 1
                                # self.log_message("unlink: " + os.path.join(dir, f))
                        os.rmdir(dir)
                    tmp = data.delete(id)
                    deleted.append(tmp)
                    if tmp:
                        self.log_message("Deleted item '%s' and %d files" % (id, files_deleted))
                except Exception as e:
                    self.log_error("%s [%s, %s]" % (str(e), dir, file))
                    # TODO: report errors?
            # Save if there was at least one deletion
            for id in deleted:
                if id is not None:
                    data.save()
                    break
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
    print('Starting server at http://%s:%d' % (server_address))
    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv

    # Disable interactive plots
    pl.ioff()

    # HACK: replace with DB connection
    data = PickleData('results-combined.pickle')

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
