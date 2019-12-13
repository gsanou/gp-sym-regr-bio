#!/usr/bin/env python

import glob
import os
import re
import warnings
import sys
from argparse import ArgumentParser
from itertools import product
from pprint import pprint
from shutil import rmtree
from tempfile import mkdtemp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory, parallel_backend
from gplearn.genetic import BaseSymbolic, SymbolicClassifier
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection import (f_classif, SelectFdr, SelectKBest,
                                       VarianceThreshold)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (auc, average_precision_score,
                             balanced_accuracy_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from tabulate import tabulate


def setup_pipe_and_param_grid():
    pipe_steps = []
    pipe_step_names = []
    pipe_props = {'has_selector': False}
    param_grid = []
    param_grid_dict = {}
    pipe_step_keys = []
    pipe_step_types = []
    for step_idx, step_keys in enumerate(args.pipe_steps):
        if any(k.title() == 'None' for k in step_keys):
            pipe_step_keys.append(
                [k for k in step_keys if k.title() != 'None'] + [None])
        else:
            pipe_step_keys.append(step_keys)
        if len(step_keys) > 1:
            pipe_step_names.append('|'.join(step_keys))
        else:
            pipe_step_names.append(step_keys[0])
    for pipe_step_combo in product(*pipe_step_keys):
        params = {}
        for step_idx, step_key in enumerate(pipe_step_combo):
            if step_key:
                if step_key in pipe_config:
                    estimator = pipe_config[step_key]['estimator']
                else:
                    run_cleanup()
                    raise RuntimeError('No pipeline config exists for {}'
                                       .format(step_key))
                if isinstance(estimator, SelectorMixin):
                    step_type = 'slr'
                    pipe_props['has_selector'] = True
                elif isinstance(estimator, TransformerMixin):
                    step_type = 'trf'
                elif isinstance(estimator, ClassifierMixin):
                    step_type = 'clf'
                elif isinstance(estimator, RegressorMixin):
                    step_type = 'rgr'
                else:
                    run_cleanup()
                    raise RuntimeError('Unsupported estimator type {}'
                                       .format(estimator))
                if step_idx < len(pipe_steps):
                    if step_type != pipe_step_types[step_idx]:
                        run_cleanup()
                        raise RuntimeError(
                            'Different step estimator types: {} {}'
                            .format(step_type, pipe_step_types[step_idx]))
                else:
                    pipe_step_types.append(step_type)
                uniq_step_name = step_type + str(step_idx)
                if 'param_grid' in pipe_config[step_key]:
                    for param, param_values in (
                            pipe_config[step_key]['param_grid'].items()):
                        if isinstance(param_values, (list, tuple, np.ndarray)):
                            if (isinstance(param_values, (list, tuple))
                                    and param_values or np.any(param_values)):
                                uniq_step_param = '{}__{}'.format(
                                    uniq_step_name, param)
                                if len(param_values) > 1:
                                    params[uniq_step_param] = param_values
                                    if uniq_step_param not in param_grid_dict:
                                        param_grid_dict[uniq_step_param] = (
                                            param_values)
                                else:
                                    estimator.set_params(
                                        **{param: param_values[0]})
                        elif param_values is not None:
                            estimator.set_params(**{param: param_values})
                if step_idx == len(pipe_steps):
                    if len(pipe_step_keys[step_idx]) > 1:
                        pipe_steps.append((uniq_step_name, None))
                    else:
                        pipe_steps.append((uniq_step_name, estimator))
                if len(pipe_step_keys[step_idx]) > 1:
                    params[uniq_step_name] = [estimator]
                    if uniq_step_name not in param_grid_dict:
                        param_grid_dict[uniq_step_name] = []
                    if estimator not in param_grid_dict[uniq_step_name]:
                        param_grid_dict[uniq_step_name].append(estimator)
            else:
                uniq_step_name = pipe_step_types[step_idx] + str(step_idx)
                params[uniq_step_name] = [None]
                if uniq_step_name not in param_grid_dict:
                    param_grid_dict[uniq_step_name] = []
                if None not in param_grid_dict[uniq_step_name]:
                    param_grid_dict[uniq_step_name].append(None)
        param_grid.append(params)
    pipe = Pipeline(pipe_steps, memory=memory)
    pipe_name = '->'.join(pipe_step_names)
    for param, param_values in param_grid_dict.items():
        if any(isinstance(v, BaseEstimator) for v in param_values):
            param_grid_dict[param] = sorted(
                ['.'.join([type(v).__module__, type(v).__qualname__])
                 if isinstance(v, BaseEstimator) else v for v in param_values],
                key=lambda x: (x is None, x))
    return (pipe, pipe_steps, pipe_name, pipe_props, param_grid,
            param_grid_dict)


def calculate_test_scores(pipe, X_test, y_test):
    scores = {}
    if hasattr(pipe, 'decision_function'):
        y_score = pipe.decision_function(X_test)
    else:
        y_score = pipe.predict_proba(X_test)[:, 1]
    for metric in args.scv_scoring:
        if metric == 'roc_auc':
            scores[metric] = roc_auc_score(y_test, y_score)
            scores['fpr'], scores['tpr'], _ = roc_curve(y_test, y_score,
                                                        pos_label=1)
        elif metric == 'balanced_accuracy':
            y_pred = pipe.predict(X_test)
            scores[metric] = balanced_accuracy_score(y_test, y_pred)
        elif metric == 'average_precision':
            scores[metric] = average_precision_score(y_test, y_score)
            scores['pre'], scores['rec'], _ = precision_recall_curve(
                y_test, y_score, pos_label=1)
            scores['pr_auc'] = auc(scores['rec'], scores['pre'])
    return scores


def add_param_cv_scores(search, param_grid_dict, param_cv_scores=None):
    if param_cv_scores is None:
        param_cv_scores = {}
    for param, param_values in param_grid_dict.items():
        if len(param_values) == 1:
            continue
        param_cv_values = search.cv_results_['param_{}'.format(param)]
        if any(isinstance(v, BaseEstimator) for v in param_cv_values):
            param_cv_values = np.array(
                ['.'.join([type(v).__module__, type(v).__qualname__])
                 if isinstance(v, BaseEstimator) else v
                 for v in param_cv_values])
        if param not in param_cv_scores:
            param_cv_scores[param] = {}
        for metric in args.scv_scoring:
            if metric not in param_cv_scores[param]:
                param_cv_scores[param][metric] = {'scores': [], 'stdev': []}
            param_metric_scores = param_cv_scores[param][metric]['scores']
            param_metric_stdev = param_cv_scores[param][metric]['stdev']
            for param_value_idx, param_value in enumerate(param_values):
                mean_cv_scores = (search.cv_results_
                                  ['mean_test_{}'.format(metric)]
                                  [param_cv_values == param_value])
                std_cv_scores = (search.cv_results_
                                 ['std_test_{}'.format(metric)]
                                 [param_cv_values == param_value])
                if param_value_idx < len(param_metric_scores):
                    param_metric_scores[param_value_idx] = np.append(
                        param_metric_scores[param_value_idx],
                        mean_cv_scores[np.argmax(mean_cv_scores)])
                    param_metric_stdev[param_value_idx] = np.append(
                        param_metric_stdev[param_value_idx],
                        std_cv_scores[np.argmax(mean_cv_scores)])
                else:
                    param_metric_scores.append(np.array(
                        [mean_cv_scores[np.argmax(mean_cv_scores)]]))
                    param_metric_stdev.append(np.array(
                        [std_cv_scores[np.argmax(mean_cv_scores)]]))
    return param_cv_scores


def plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                          param_cv_scores):
    sns.set_palette(sns.color_palette('hls', len(args.scv_scoring)))
    for param in param_cv_scores:
        mean_cv_scores, std_cv_scores = {}, {}
        for metric in args.scv_scoring:
            param_metric_scores = param_cv_scores[param][metric]['scores']
            param_metric_stdev = param_cv_scores[param][metric]['stdev']
            if any(len(l) > 1 for l in param_metric_scores):
                mean_cv_scores[metric], std_cv_scores[metric] = [], []
                for param_value_scores in param_metric_scores:
                    mean_cv_scores[metric].append(np.mean(param_value_scores))
                    std_cv_scores[metric].append(np.std(param_value_scores))
            else:
                mean_cv_scores[metric] = np.ravel(param_metric_scores)
                std_cv_scores[metric] = np.ravel(param_metric_stdev)
        plt.figure()
        param_type = re.sub(r'^([a-z]+)\d+', r'\1', param, count=1)
        if param_type in params_num_xticks:
            x_axis = param_grid_dict[param]
            plt.xticks(x_axis)
        elif param_type in params_fixed_xticks:
            x_axis = range(len(param_grid_dict[param]))
            xtick_labels = [v.split('.')[-1]
                            if param_type in pipeline_step_types
                            and not args.long_label_names
                            and v is not None else str(v)
                            for v in param_grid_dict[param]]
            plt.xticks(x_axis, xtick_labels)
        else:
            raise RuntimeError('No ticks config exists for {}'
                               .format(param_type))
        plt.xlim([min(x_axis), max(x_axis)])
        plt.title('{}\n{}\nEffect of {} on CV Performance Metrics'.format(
            dataset_name, pipe_name, param), fontsize=args.title_font_size)
        plt.xlabel(param, fontsize=args.axis_font_size)
        plt.ylabel('CV Score', fontsize=args.axis_font_size)
        for metric_idx, metric in enumerate(args.scv_scoring):
            plt.plot(x_axis, mean_cv_scores[metric], lw=2, alpha=0.8,
                     label='Mean {}'.format(metric_label[metric]))
            plt.fill_between(x_axis,
                             [m - s for m, s in zip(mean_cv_scores[metric],
                                                    std_cv_scores[metric])],
                             [m + s for m, s in zip(mean_cv_scores[metric],
                                                    std_cv_scores[metric])],
                             alpha=0.2, color='grey', label=(
                                 r'$\pm$ 1 std. dev.'
                                 if metric_idx == len(args.scv_scoring) - 1
                                 else None))
        plt.legend(loc='lower right', fontsize='medium')
        plt.tick_params(labelsize=args.axis_font_size)
        plt.grid(True, alpha=0.3)


def run_model_selection():
    pipe, pipe_steps, pipe_name, pipe_props, param_grid, param_grid_dict = (
        setup_pipe_and_param_grid())
    dataset_name, data_file_ext = os.path.splitext(
        os.path.split(args.train_dataset)[1])
    meta_file_ext = os.path.splitext(os.path.split(args.train_meta)[1])[1]
    if os.path.isfile(args.train_dataset) and data_file_ext == '.csv':
        data = pd.read_csv(args.train_dataset, index_col=0)
        if os.path.isfile(args.train_meta) and meta_file_ext == '.csv':
            sample_meta = pd.read_csv(args.train_meta, index_col=0)
            X = np.array(data.T)
            y = pd.factorize(sample_meta['Class'])[0]
            feature_meta = pd.read_csv('data/ensg_symbol_grch38p2_gtfv22.tsv',
                                       keep_default_na=False, index_col=0,
                                       sep='\t')
        else:
            raise IOError('File does not exist/invalid: {}'
                          .format(args.train_meta))
    else:
        raise IOError('File does not exist/invalid: {}'
                      .format(args.train_dataset))
    cv_splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                                  random_state=args.random_seed)
    search = GridSearchCV(
        pipe, cv=cv_splitter, error_score=0, iid=False, n_jobs=args.n_jobs,
        param_grid=param_grid, refit=args.scv_refit, return_train_score=False,
        scoring=args.scv_scoring, verbose=args.scv_verbose)
    if args.verbose > 0:
        print('{}:'.format(type(search).__name__))
        pprint({k: vars(v) if k == 'estimator' else v
                for k, v in vars(search).items()})
    if args.verbose > 1 and param_grid_dict:
        print('Param grid dict:')
        pprint(param_grid_dict)
    if args.verbose > 0 or args.scv_verbose > 0:
        print('Train:' if args.test_dataset else 'Dataset:', dataset_name,
              X.shape)
    if args.load_only:
        run_cleanup()
        sys.exit()
    # train-test nested cv
    if not args.test_dataset:
        split_results = []
        param_cv_scores = {}
        test_splitter = StratifiedKFold(n_splits=args.test_folds,
                                        shuffle=True,
                                        random_state=args.random_seed)
        for split_idx, (train_idxs, test_idxs) in enumerate(
                test_splitter.split(X, y)):
            with parallel_backend(args.parallel_backend):
                search.fit(X[train_idxs], y[train_idxs])
            feature_idxs = np.arange(X[train_idxs].shape[1])
            for step in search.best_estimator_.named_steps:
                if hasattr(search.best_estimator_.named_steps[step],
                           'get_support'):
                    feature_idxs = feature_idxs[
                        search.best_estimator_.named_steps[step].get_support()]
            feature_weights = np.zeros_like(feature_idxs, dtype=float)
            final_estimator = search.best_estimator_.steps[-1][1]
            if hasattr(final_estimator, 'coef_'):
                feature_weights = np.square(final_estimator.coef_[0])
            param_cv_scores = add_param_cv_scores(search, param_grid_dict,
                                                  param_cv_scores)
            split_scores = {'cv': {}}
            for metric in args.scv_scoring:
                split_scores['cv'][metric] = (search.cv_results_
                                              ['mean_test_{}'.format(metric)]
                                              [search.best_index_])
            split_scores['te'] = calculate_test_scores(
                search.best_estimator_, X[test_idxs], y[test_idxs])
            if args.verbose > 0:
                print('Dataset:', dataset_name, ' Split: {:>{width}d}'
                      .format(split_idx + 1,
                              width=len(str(args.test_folds))), end=' ')
                for metric in args.scv_scoring:
                    print(' {} (CV / Test): {:.4f} / {:.4f}'.format(
                        metric_label[metric], split_scores['cv'][metric],
                        split_scores['te'][metric]), end=' ')
                    if metric == 'average_precision':
                        print(' PR AUC Test: {:.4f}'.format(
                            split_scores['te']['pr_auc']), end=' ')
                print(' Params:', {
                    k: ('.'.join([type(v).__module__, type(v).__qualname__])
                        if isinstance(v, BaseEstimator) else v)
                    for k, v in search.best_params_.items()}, end=' ')
                if isinstance(search.best_estimator_.steps[-1][1],
                              BaseSymbolic):
                    print(' Program:', search.best_estimator_.steps[-1][1])
                else:
                    print()
                selected_feature_meta = feature_meta.iloc[feature_idxs].copy()
                if np.any(feature_weights):
                    selected_feature_meta['Weight'] = feature_weights
                    print('Feature Ranking:')
                    print(tabulate(selected_feature_meta.sort_values(
                        by='Weight', ascending=False), floatfmt='.6e',
                                   headers='keys'))
                else:
                    print('Features:')
                    print(tabulate(selected_feature_meta, headers='keys'))
            split_results.append({
                'feature_idxs': feature_idxs,
                'feature_weights': feature_weights,
                'scores': split_scores})

            import graphviz
            graph = graphviz.Source(search.best_estimator_.steps[-1][1]
                                    ._program.export_graphviz())
            graph.render('test.gv')

            if args.pipe_memory:
                memory.clear(warn=False)
        scores = {'cv': {}, 'te': {}}
        num_features = []
        for split_result in split_results:
            for metric in args.scv_scoring:
                if metric not in scores['cv']:
                    scores['cv'][metric] = []
                    scores['te'][metric] = []
                scores['cv'][metric].append(
                    split_result['scores']['cv'][metric])
                scores['te'][metric].append(
                    split_result['scores']['te'][metric])
                if metric == 'average_precision':
                    if 'pr_auc' not in scores['te']:
                        scores['te']['pr_auc'] = []
                    scores['te']['pr_auc'].append(
                        split_result['scores']['te']['pr_auc'])
            num_features.append(split_result['feature_idxs'].size)
        print('Dataset:', dataset_name, X.shape, end=' ')
        for metric in args.scv_scoring:
            print(' Mean {} (CV / Test): {:.4f} / {:.4f}'.format(
                metric_label[metric], np.mean(scores['cv'][metric]),
                np.mean(scores['te'][metric])), end=' ')
            if metric == 'average_precision':
                print(' Mean PR AUC Test: {:.4f}'.format(
                    np.mean(scores['te']['pr_auc'])), end=' ')
        if num_features and pipe_props['has_selector']:
            print(' Mean Features: {:.0f}'.format(np.mean(num_features)))
        else:
            print()
        # calculate overall feature ranking
        feature_idxs = []
        for split_result in split_results:
            feature_idxs.extend(split_result['feature_idxs'])
        feature_idxs = sorted(list(set(feature_idxs)))
        feature_matrix_idx = {}
        for idx, feature_idx in enumerate(feature_idxs):
            feature_matrix_idx[feature_idx] = idx
        weights_matrix = np.zeros(
            (len(feature_idxs), len(split_results)), dtype=float)
        scores_cv_matrix = {}
        for metric in args.scv_scoring:
            scores_cv_matrix[metric] = np.zeros(
                (len(feature_idxs), len(split_results)), dtype=float)
        for split_idx, split_result in enumerate(split_results):
            for idx, feature_idx in enumerate(split_result['feature_idxs']):
                (weights_matrix[feature_matrix_idx[feature_idx]]
                 [split_idx]) = split_result['feature_weights'][idx]
                for metric in args.scv_scoring:
                    (scores_cv_matrix[metric]
                     [feature_matrix_idx[feature_idx]][split_idx]) = (
                         split_result['scores']['cv'][metric])
        feature_mean_weights, feature_mean_scores = [], []
        for idx in range(len(feature_idxs)):
            feature_mean_weights.append(np.mean(weights_matrix[idx]))
            feature_mean_scores.append(np.mean(
                scores_cv_matrix[args.scv_refit][idx]))
        if args.verbose > 0:
            selected_feature_meta = feature_meta.iloc[feature_idxs].copy()
            if np.any(feature_mean_weights):
                selected_feature_meta['Mean Weight'] = feature_mean_weights
                print('Overall Feature Ranking:')
                print(tabulate(selected_feature_meta.sort_values(
                    by='Mean Weight', ascending=False), floatfmt='.6e',
                               headers='keys'))
            else:
                print('Overall Features:')
                print(tabulate(selected_feature_meta, headers='keys'))
        plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                              param_cv_scores)
        # plot roc and pr curves
        if 'roc_auc' in args.scv_scoring:
            sns.set_palette(sns.color_palette('hls', 2))
            plt.figure()
            plt.title('{}\n{}\nROC Curve'.format(
                dataset_name, pipe_name), fontsize=args.title_font_size)
            plt.xlabel('False Positive Rate', fontsize=args.axis_font_size)
            plt.ylabel('True Positive Rate', fontsize=args.axis_font_size)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            tprs = []
            mean_fpr = np.linspace(0, 1, 100)
            for split_result in split_results:
                tprs.append(np.interp(mean_fpr,
                                      split_result['scores']['te']['fpr'],
                                      split_result['scores']['te']['tpr']))
                tprs[-1][0] = 0.0
                plt.plot(split_result['scores']['te']['fpr'],
                         split_result['scores']['te']['tpr'], alpha=0.2,
                         color='darkgrey', lw=1)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_roc_auc = np.mean(scores['te']['roc_auc'])
            std_roc_auc = np.std(scores['te']['roc_auc'])
            mean_num_features = np.mean(num_features)
            std_num_features = np.std(num_features)
            plt.plot(mean_fpr, mean_tpr, lw=3, alpha=0.8, label=(
                r'Test Mean ROC (AUC = {:.4f} $\pm$ {:.2f}, '
                r'Features = {:.0f} $\pm$ {:.0f})').format(
                    mean_roc_auc, std_roc_auc, mean_num_features,
                    std_num_features))
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2,
                             color='grey', label=r'$\pm$ 1 std. dev.')
            plt.plot([0, 1], [0, 1], alpha=0.2, color='grey',
                     linestyle='--', lw=3, label='Chance')
            plt.legend(loc='lower right', fontsize='medium')
            plt.tick_params(labelsize=args.axis_font_size)
            plt.grid(False)
        if 'average_precision' in args.scv_scoring:
            sns.set_palette(sns.color_palette('hls', 10))
            plt.figure()
            plt.title('{}\n{}\nPR Curve'.format(
                dataset_name, pipe_name), fontsize=args.title_font_size)
            plt.xlabel('Recall', fontsize=args.axis_font_size)
            plt.ylabel('Precision', fontsize=args.axis_font_size)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            pres, scores['te']['pr_auc'] = [], []
            mean_rec = np.linspace(0, 1, 100)
            for split_result in split_results:
                scores['te']['pr_auc'].append(
                    split_result['scores']['te']['pr_auc'])
                pres.append(np.interp(mean_rec,
                                      split_result['scores']['te']['rec'],
                                      split_result['scores']['te']['pre']))
                pres[-1][0] = 1.0
                plt.step(split_result['scores']['te']['rec'],
                         split_result['scores']['te']['pre'], alpha=0.2,
                         color='darkgrey', lw=1, where='post')
            mean_pre = np.mean(pres, axis=0)
            mean_pre[-1] = 0.0
            mean_pr_auc = np.mean(scores['te']['pr_auc'])
            std_pr_auc = np.std(scores['te']['pr_auc'])
            mean_num_features = np.mean(num_features)
            std_num_features = np.std(num_features)
            plt.step(mean_rec, mean_pre, alpha=0.8, lw=3, where='post',
                     label=(r'Test Mean PR (AUC = {:.4f} $\pm$ {:.2f}, '
                            r'Features = {:.0f} $\pm$ {:.0f})').format(
                                mean_pr_auc, std_pr_auc, mean_num_features,
                                std_num_features))
            std_pre = np.std(pres, axis=0)
            pres_upper = np.minimum(mean_pre + std_pre, 1)
            pres_lower = np.maximum(mean_pre - std_pre, 0)
            plt.fill_between(mean_rec, pres_lower, pres_upper, alpha=0.2,
                             color='grey', label=r'$\pm$ 1 std. dev.')
            plt.legend(loc='lower right', fontsize='medium')
            plt.tick_params(labelsize=args.axis_font_size)
            plt.grid(False)


def shifted_log2(X, shift=1):
    return np.log2(X + shift)


def run_cleanup():
    if args.pipe_memory:
        rmtree(cachedir)
    if glob.glob('/tmp/Rtmp*'):
        for rtmp in glob.glob('/tmp/Rtmp*'):
            rmtree(rtmp)


def str_list(arg):
    return list(map(str, arg.split(',')))


parser = ArgumentParser()
parser.add_argument('--train-dataset', '--dataset', '--train-data',
                    type=str, required=True, help='training dataset')
parser.add_argument('--train-meta', type=str, required=True,
                    help='training metadata')
parser.add_argument('--pipe-steps', type=str_list, nargs='+', required=True,
                    help='pipeline step names')
parser.add_argument('--test-dataset', '--test-data', type=str,
                    help='test dataset')
parser.add_argument('--test-meta', type=str,
                    help='test metadata')
parser.add_argument('--slr-fdr-a', type=float, nargs='+',
                    help='slr fdr alpha')
parser.add_argument('--slr-skb-k', type=int, nargs='+',
                    help='slr skb k')
parser.add_argument('--slr-skb-k-min', type=int, default=1,
                    help='slr skb k min')
parser.add_argument('--slr-skb-k-max', type=int,
                    help='slr skb k max')
parser.add_argument('--slr-skb-k-step', type=int, default=1,
                    help='slr skb k step')
parser.add_argument('--clf-svm-c', type=float, nargs='+',
                    help='clf svm c')
parser.add_argument('--clf-svm-c-min', type=float,
                    help='clf svm c min')
parser.add_argument('--clf-svm-c-max', type=float,
                    help='clf svm c max')
parser.add_argument('--clf-svm-kern', type=str, nargs='+',
                    help='clf svm kernel')
parser.add_argument('--clf-svm-deg', type=int, nargs='+',
                    help='clf svm poly degree')
parser.add_argument('--clf-svm-g', type=str, nargs='+',
                    help='clf svm gamma')
parser.add_argument('--clf-svm-cache', type=int, default=2000,
                    help='libsvm cache size')
parser.add_argument('--clf-sgd-a', type=float, nargs='+',
                    help='clf sgd alpha')
parser.add_argument('--clf-sgd-a-min', type=float,
                    help='clf sgd alpha min')
parser.add_argument('--clf-sgd-a-max', type=float,
                    help='clf sgd alpha max')
parser.add_argument('--clf-sgd-loss', type=str, nargs='+',
                    choices=['hinge', 'log', 'modified_huber', 'squared_hinge',
                             'perceptron', 'squared_loss', 'huber',
                             'epsilon_insensitive',
                             'squared_epsilon_insensitive'],
                    help='clf sgd loss')
parser.add_argument('--clf-sgd-penalty', type=str,
                    choices=['l1', 'l2', 'elasticnet'], default='l2',
                    help='clf sgd penalty')
parser.add_argument('--clf-sgd-l1r', type=float, nargs='+',
                    help='clf sgd l1 ratio')
parser.add_argument('--clf-sym-fs', type=str_list, nargs='+',
                    help='clf sym function set')
parser.add_argument('--clf-sym-g', type=int, nargs='+',
                    help='clf sym generations')
parser.add_argument('--clf-sym-pcr', type=float, nargs='+',
                    help='clf sym p crossover')
parser.add_argument('--clf-sym-phm', type=float, nargs='+',
                    help='clf sym p hoist mutation')
parser.add_argument('--clf-sym-ppm', type=float, nargs='+',
                    help='clf sym p point mutation')
parser.add_argument('--clf-sym-ppr', type=float, nargs='+',
                    help='clf sym p point replace')
parser.add_argument('--clf-sym-psm', type=float, nargs='+',
                    help='clf sym p subtree mutation')
parser.add_argument('--clf-sym-ps', type=int, nargs='+',
                    help='clf sym population size')
parser.add_argument('--clf-sym-ts', type=int, nargs='+',
                    help='clf sym tournament size')
parser.add_argument('--cv-folds', type=int, default=5,
                    help='cv folds')
parser.add_argument('--scv-verbose', type=int,
                    help='scv verbosity')
parser.add_argument('--scv-scoring', type=str, nargs='+',
                    choices=['roc_auc', 'balanced_accuracy',
                             'average_precision'],
                    default=['roc_auc', 'balanced_accuracy',
                             'average_precision'],
                    help='scv scoring metric')
parser.add_argument('--scv-refit', type=str, default='roc_auc',
                    choices=['roc_auc', 'balanced_accuracy',
                             'average_precision'],
                    help='scv refit scoring metric')
parser.add_argument('--test-folds', type=int, default=5,
                    help='num outer folds')
parser.add_argument('--param-cv-score-meth', type=str,
                    choices=['best', 'all'], default='best',
                    help='param cv scores calculation method')
parser.add_argument('--title-font-size', type=int, default=14,
                    help='figure title font size')
parser.add_argument('--axis-font-size', type=int, default=14,
                    help='figure axis font size')
parser.add_argument('--long-label-names', default=False, action='store_true',
                    help='figure long label names')
parser.add_argument('--n-jobs', type=int, default=-1,
                    help='num parallel jobs')
parser.add_argument('--parallel-backend', type=str, default='loky',
                    help='joblib parallel backend')
parser.add_argument('--pipe-memory', default=False, action='store_true',
                    help='turn on pipeline memory')
parser.add_argument('--cache-dir', type=str, default='/tmp',
                    help='cache dir')
parser.add_argument('--random-seed', type=int, default=777,
                    help='random state seed')
parser.add_argument('--verbose', type=int, default=1,
                    help='program verbosity')
parser.add_argument('--load-only', default=False, action='store_true',
                    help='set up model selection and load dataset only')
args = parser.parse_args()

# suppress linux conda qt5 wayland warning
if sys.platform.startswith('linux'):
    os.environ['XDG_SESSION_TYPE'] = 'x11'

if args.scv_verbose is None:
    args.scv_verbose = args.verbose
if args.parallel_backend == 'multiprocessing':
    # ignore gplearn invalid value warnings (when parsimony_coefficient='auto')
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message=('^invalid value encountered in '
                                     'double_scalars'),
                            module='gplearn.genetic')
    # ignore joblib peristence time warnings
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='^Persisting input arguments took',
                            module='sklearn.pipeline')
else:
    python_warnings = ([os.environ['PYTHONWARNINGS']]
                       if 'PYTHONWARNINGS' in os.environ else [])
    python_warnings.append(
        'ignore:invalid value encountered in double_scalars:'
        'RuntimeWarning:gplearn.genetic')
    python_warnings.append(
        'ignore:Persisting input arguments took:'
        'UserWarning:sklearn.pipeline')
    os.environ['PYTHONWARNINGS'] = ','.join(python_warnings)

if args.pipe_memory:
    cachedir = mkdtemp(dir='/tmp')
    memory = Memory(location=cachedir, verbose=0)
else:
    memory = None

pipeline_step_types = ('slr', 'trf', 'clf', 'rgr')
cv_params = {k: v for k, v in vars(args).items()
             if k.startswith(pipeline_step_types)}
for cv_param, cv_param_values in cv_params.items():
    if cv_param_values is None:
        continue
    if cv_param in ('slr_skb_k', 'slr_fdr_a', 'clf_svm_c', 'clf_svm_kern',
                    'clf_svm_deg', 'clf_svm_g', 'clf_sgd_a', 'clf_sgd_loss',
                    'clf_sgd_l1r', 'clf_sym_fs', 'clf_sym_g', 'clf_sym_pcr',
                    'clf_sym_phm', 'clf_sym_ppm', 'clf_sym_ppr', 'clf_sym_psm',
                    'clf_sym_ps', 'clf_sym_ts'):
        cv_params[cv_param] = sorted(cv_param_values)
    elif cv_param == 'slr_skb_k_max':
        if cv_params['slr_skb_k_min'] == 1 and cv_params['slr_skb_k_step'] > 1:
            cv_params['slr_skb_k'] = [1] + list(range(
                0, cv_params['slr_skb_k_max'] + cv_params['slr_skb_k_step'],
                cv_params['slr_skb_k_step']))[1:]
        else:
            cv_params['slr_skb_k'] = list(range(
                cv_params['slr_skb_k_min'],
                cv_params['slr_skb_k_max'] + cv_params['slr_skb_k_step'],
                cv_params['slr_skb_k_step']))
    elif cv_param == 'clf_svm_c_max':
        log_start = int(np.floor(np.log10(abs(cv_params['clf_svm_c_min']))))
        log_end = int(np.floor(np.log10(abs(cv_params['clf_svm_c_max']))))
        cv_params['clf_svm_c'] = np.logspace(log_start, log_end,
                                             log_end - log_start + 1)
    elif cv_param == 'clf_sgd_a_max':
        log_start = int(np.floor(np.log10(abs(cv_params['clf_sgd_a_min']))))
        log_end = int(np.floor(np.log10(abs(cv_params['clf_sgd_a_max']))))
        cv_params['clf_sgd_a'] = np.logspace(log_start, log_end,
                                             log_end - log_start + 1)

pipe_config = {
    'ShiftedLog2Transformer': {
        'estimator':  FunctionTransformer(shifted_log2, check_inverse=False,
                                          validate=True)},
    'VarianceThreshold': {
        'estimator':  VarianceThreshold()},
    'SelectFdr-ANOVAFScorer': {
        'estimator': SelectFdr(score_func=f_classif),
        'param_grid': {
            'alpha': cv_params['slr_fdr_a']}},
    'SelectKBest-ANOVAFScorer': {
        'estimator': SelectKBest(score_func=f_classif),
        'param_grid': {
            'k': cv_params['slr_skb_k']}},
    'StandardScaler': {
        'estimator': StandardScaler()},
    'SVC': {
        'estimator': SVC(cache_size=args.clf_svm_cache,
                         class_weight='balanced', gamma='scale',
                         random_state=args.random_seed),
        'param_grid': {
            'C': cv_params['clf_svm_c'],
            'kernel': cv_params['clf_svm_kern'],
            'degree': cv_params['clf_svm_deg'],
            'gamma': cv_params['clf_svm_g']}},
    'SGDClassifier': {
        'estimator': SGDClassifier(class_weight='balanced',
                                   penalty=args.clf_sgd_penalty,
                                   random_state=args.random_seed),
        'param_grid': {
            'alpha': cv_params['clf_sgd_a'],
            'loss': cv_params['clf_sgd_loss'],
            'l1_ratio': cv_params['clf_sgd_l1r']}},
    'SymbolicClassifier': {
        'estimator': SymbolicClassifier(parsimony_coefficient='auto',
                                        random_state=args.random_seed,
                                        stopping_criteria=0.01),
        'param_grid': {
            'function_set': cv_params['clf_sym_fs'],
            'generations': cv_params['clf_sym_g'],
            'p_crossover': cv_params['clf_sym_pcr'],
            'p_hoist_mutation': cv_params['clf_sym_phm'],
            'p_point_mutation': cv_params['clf_sym_ppm'],
            'p_point_replace': cv_params['clf_sym_ppr'],
            'p_subtree_mutation': cv_params['clf_sym_psm'],
            'population_size': cv_params['clf_sym_ps'],
            'tournament_size': cv_params['clf_sym_ts']}}}

params_num_xticks = [
    'slr__k',
    'clf__degree',
    'clf__generations',
    'clf__l1_ratio',
    'clf__population_size',
    'clf__p_crossover',
    'clf__p_hoist_mutation',
    'clf__p_point_mutation',
    'clf__p_point_replace',
    'clf__p_subtree_mutation',
    'clf__tournament_size']
params_fixed_xticks = [
    'slr__alpha',
    'clf',
    'clf__alpha',
    'clf__C',
    'clf__function_set',
    'clf__gamma',
    'clf__kernel',
    'clf__loss']
metric_label = {
    'roc_auc': 'ROC AUC',
    'balanced_accuracy': 'BCR',
    'average_precision': 'AVG PRE'}

run_model_selection()
run_cleanup()
plt.show()
