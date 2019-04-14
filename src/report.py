#!/usr/bin/env python3

import json
import io
import os
import importlib
import binascii
import matplotlib.pyplot as pl
from math import floor, log10
from plots import plot_one, plot_many, plot_mesh
from incense import ExperimentLoader
from common import generate_mesh, make_filename
from dolfin import Mesh
from typing import Union, List
from flask import Flask, render_template, make_response, Response


class MongoData(object):
    """ A simple interface around the mongo database of results.
    FIXME: This is naive and loads all data at once!
    """
    def __init__(self, mongo_uri: str, db_name: str):
        self._loader = ExperimentLoader(mongo_uri=mongo_uri, db_name=db_name)
        self._experiments = None
        self.load()

    def load(self):
        self._loader.cache_clear()
        self._experiments = self._loader.find({})

    def __getitem__(self, exp_id: Union[str, int]):
        # Should be cached by incense...
        return self._loader.find_by_id(int(exp_id))

    def columns(self):
        """ Returns a json string with the colum data for footable. """
        cols = [{'name': 'exp_name', 'title': 'Experiment'},
                {'name': 'form_name', 'title': 'Q2'},
                {'name': 'init', 'title': 'Initial data'},
                {'name': 'theta', 'title': 'theta', 'type': 'number'},
                {'name': 'mu', 'title': 'mu', 'type': 'number'},
                {'name': 'hmin_power', 'title': 'h^', 'type': 'number'},
                {'name': 'e_stop', 'title': 'eps', 'type': 'number'},
                # {'name': 'steps', 'title': 'Steps', 'type': 'number', 'filterable': False},
                {'name': 'time', 'title': 'Duration', 'type': 'time', 'filterable': False},
                {'name': 'plot', 'title': '', 'type': 'html', 'filterable': False},
                {'name': 'results', 'type': 'html', 'breakpoints': 'all',
                 'title': 'Results file:', 'filterable': False},
                {'name': 'mesh', 'title': 'Mesh file:', 'breakpoints': 'all'},
                {'name': 'select', 'title': '', 'filterable': False}]
        return json.dumps(cols)

    def rows(self):
        """ Returns a json dict with the row data for footable. """
        def toggle_button(exp_id: str) -> str:
            form_input = '<input class="tgl tgl-flat" id="%d" type="checkbox"/>' % exp_id
            form_label = '<label class="tgl-btn" for="%d"></label>' % exp_id
            return form_input + form_label

        ret = []
        for e in self._experiments:
            conf = e.config
            mesh_file = generate_mesh(conf.mesh_type, conf.mesh_m, conf.mesh_n)
            msh = Mesh(mesh_file)
            mu = conf.mu_scale / msh.hmin()**conf.hmin_power
            file_name = make_filename(e.experiment.name, conf.theta, mu, makedir=False)
            
            row_data = {}
            row_data['exp_name'] = e.experiment.name
            row_data['form_name'] = conf.qform[:4]
            row_data['init'] = conf.init
            # row_data['steps'] = e....?
            row_data['theta'] = round(conf.theta, 3)
            row_data['mu'] = round(mu, 2)
            row_data['hmin_power'] = round(conf.hmin_power, 1)
            row_data['e_stop'] = int(log10(conf.e_stop_mult)) - 1
            try:
                tt = (e.stop_time - e.start_time).seconds
                hh = floor(tt / 3600)
                mm = floor((tt - hh * 3600) / 60)
                ss = floor((tt - hh * 3600 - mm * 60))
                row_data['time'] = "%02d:%02d:%02d" % (hh, mm, ss)
            except AttributeError:
                row_data['time'] = e.status.lower()
            row_data['plot'] = '<a class="plot_one" href="#single_plot_target" ' \
                               + 'data-value="%d" onclick="show_one(this)">Plot</a>' % e.id
            row_data['results'] = '<a href="pvd://%s">%s</a>' % \
                                  (file_name, os.path.basename(file_name))
            row_data['mesh'] = '<a class="plot_one" href="#single_plot_target" ' \
                               + 'data-value="%s" onclick="show_mesh(this)">%s</a>' % \
                               (mesh_file, os.path.basename(mesh_file))
            row_data['select'] = toggle_button(e.id)
            ret.append(row_data)

        return json.dumps(ret)


# Disable interactive plots
pl.ioff()

data = MongoData('mongo:27017', 'lvk')
app = Flask(__name__)


def set_image_headers(response: Response):
    response.headers['Pragma'] = 'public'
    response.headers['Pragma-directive'] = "no-cache"
    response.headers['Cache-directive'] = "no-cache"
    response.headers['Cache-control'] = "no-cache"
    response.headers['Pragma'] = "no-cache"
    response.headers['Expires'] = "0"
    # response.headers['Expires'] = gmdate('D, d M Y H:i:s \G\M\T', time() + 86400))
    response.mimetype = 'image/png'

    
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/rows')
def get_rows():
    response = make_response(data.rows(), 200)
    response.mimetype = 'application/json'
    return response


@app.route('/api/columns')
def get_columns():
    response = make_response(data.columns(), 200)
    response.mimetype = 'application/json'
    return response


def prepare_plot(b64: bool=False):
    with io.BytesIO() as buf:
        pl.savefig(buf, format='png')
        pl.close()
        buf.seek(0)
        if b64:
            buf = io.BytesIO(binascii.b2a_base64(buf.getvalue()))
        response = make_response(buf.getvalue())
        set_image_headers(response)
    return response


@app.route('/api/plot_one/<int:exp_id>')
def get_single_plot(exp_id: int):
    try:
        e = data[exp_id]
        metrics = e.metrics
        metrics.update({'init': e.config.init, 'theta': e.config.theta,
                        'steps': len(metrics['J'])})
        plot_one(metrics, None, None)
        response = prepare_plot()
    except IndexError as e:
        out = render_template('error.html', type=400,
                              msg='Run %d does not exist' % exp_id)
        response = make_response(out, 400)
    except Exception as e:
        response = make_response(render_template('error.html', type=400,
                                                 msg=str(e)),
                                 400)
    return response


@app.route('/api/plot_multiple/<string:experiment_ids>')
def get_multiple_plots(experiment_ids: str):
    experiments = [data[int(exp_id)] for exp_id in experiment_ids.split(',')]
    try:
        metrics = [e.metrics for e in experiments]
        for e, m in zip(experiments, metrics):
            m.update({'init': e.config.init, 'theta': e.config.theta,
                      'steps': len(m['J'])})
        plot_many(metrics, None, None)
        response = prepare_plot(b64=True)
    except IndexError as e:
        out = render_template('error.html', type=400,
                              msg='Run "%s" does not exist' % key)
        response = make_response(out, 400)
    except Exception as e:
        response = make_response(render_template('error.html', type=400,
                                                 msg=str(e)),
                                 400)
    return response


@app.route('/api/meshes/<string:mesh_file>')
def get_mesh_plot(mesh_file: str):
    try:
        # FIXME: should use generate_mesh() here too
        plot_mesh(os.path.join('..', 'meshes', mesh_file))
        response = prepare_plot()
    except IndexError as e:
        out = render_template('error.html', type=400,
                              msg='Mesh "%s" does not exist' % mesh_file)
        response = make_response(out, 400)
    except Exception as e:
        response = make_response(render_template('error.html', type=400,
                                                 msg=str(e)), 400)
    return response


@app.route('/api/reload')
def reload():
    data.load()
    # quite redundant, but jQuery expects something
    response = make_response(json.dumps({'command':'reload', 'status': 'ok'}), 200)
    response.mimetype = 'application/json'
    return response


@app.route('/api/delete/<string:experiment_ids>')
def delete(experiment_ids: str):
    experiment_ids = list(map(int, experiment_ids.split(',')))
    deleted = []
    for e in [data[exp_id] for exp_id in experiment_ids]:
        files_deleted = 0
        msh = Mesh(generate_mesh(e.config.mesh_type, e.config.mesh_m,
                                 e.config.mesh_n))
        mu = e.config.mu_scale / msh.hmin()**e.config.hmin_power
        filename = make_filename(e.experiment.name, e.config.theta,
                                 mu, makedir=False)
        dir, file = os.path.split(filename)
        try:
            # Be extra careful: a "safe" path to delete is not
            # absolute and contains only vtu and pvd files
            if not os.path.isabs(dir) and dir != "":
                dir = os.path.join(os.getcwd(), dir)
                for f in os.listdir(dir):
                    _, ext = os.path.splitext(f)
                    if ext in ('.pvd', '.vtu'):
                        os.unlink(os.path.join(dir, f))
                        files_deleted += 1
                        # self.log_message("unlink: %s" % os.path.join(dir, f))
                if files_deleted > 0:
                    os.rmdir(dir)
            tmp = data.delete(id)
            if tmp:
                deleted.append(tmp)
                app.logger.info("Deleted item '%s' and %d files" %
                                (id, files_deleted))
        except Exception as e:
            app.logger.error("%s [%s, %s]" % (str(e), dir, file))
            # TODO: report errors?

    # Save if there was at least one deletion
    if deleted:
        data.save()
    response = make_response(json.dumps(deleted), 200)
    response.mimetype = 'application/json'
    return response


@app.errorhandler(404)
def page_not_found(e):
  return render_template('error.html', type=404, msg="Page not found"), 404
