from collections import namedtuple
import numpy as np
import scipy as sp
import scipy.interpolate
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SimpleInterpolator(BaseEstimator):
    def __init__(self, kind='linear', fill_value=0.0):
        self.kind = kind
        self.fill_value = fill_value
        self.interpolator_ = None

    # --- public functions --
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.kind == 'linear':
            self.interpolator_ = sp.interpolate.LinearNDInterpolator(points=X,
                                                                     values=y,
                                                                     fill_value=self.fill_value,
                                                                     rescale=False)
        elif self.kind == 'nearest':
            self.interpolator_ = sp.interpolate.NearestNDInterpolator(x=X,
                                                                      y=y,
                                                                      rescale=False)
        else:
            raise NotImplementedError

    def predict(self, X):
        check_is_fitted(self, ["interpolator_"])
        X = check_array(X)
        return self.interpolator_(X)


Lesson = namedtuple('Lesson', ['X', 'F', 'y'])


class Teacher:
    """
    Class for online learning guidance

    Parameters
    ----------
    tol : float, default: 0.02
        Tolerance for stopping the lesson.

    criterion : string, 'mae' or 'ae', default: 'mae'
        Specifies whether a MAE or AE is used to determine the lesson's end.

    """

    def __init__(self, sim, pupils, tol=0.02, criterion='mae', max_iter=np.inf, shuffle=False,
                 guidance=False, lesson_size=np.inf, seed=42, plotter=None, specify_end=False,
                 fitness_func=None, fitness_threshold=1.5, **kwargs):

        self.sim = sim
        self.pupils = pupils
        self.tol = tol
        self.criterion = criterion.lower()
        self.max_iter = max_iter
        self.shuffle = shuffle
        np.random.seed(seed)
        self.guidance=guidance
        self.lesson_size=lesson_size
        self.specify_end = specify_end
        self.fitness_func=fitness_func
        self.fitness_threshold = fitness_threshold
        self.kwargs = kwargs

        self.plotter = plotter
        self.history = dict(error=[],
                            model_complexity=[],
                            activity=[],
                            pred=np.array([]))

    def _sim(self, prediction, X_pred):
        for idx, p in enumerate(prediction):
            if self.specify_end:
                res = self.sim.solve(p[0:2], start_end=(0.0, p[2]), use_sp2c=True)
            else:
                res = self.sim.solve(p, use_sp2c=True)
            X_pred[idx, :] = res[0][-1, :X_pred.shape[1]]

    def _evaluate_performance(self, prediction, training_content):

        if self.fitness_func:
            X_pred = np.zeros((prediction.shape[0], 4))
            F = np.zeros((prediction.shape[0], 1))
            for i, p in enumerate(prediction):
                F[i], X_pred[i, :] = self.fitness_func(p)

        else:
            X_pred = np.zeros_like(training_content)
            self._sim(prediction, X_pred)

        AE = np.abs(training_content - X_pred)
        if self.criterion == 'mae':
            error = np.mean(AE[:, 0])
        elif self.criterion == 'ae':
            error = np.max(AE[:, 0])
        elif self.criterion == 'rmse':
            error = np.sqrt(np.mean((training_content - X_pred)[:, 0]**2))
        else:
            raise NotImplementedError

        self.history['error'].append(error)
        self.history['activity'].append(AE[:, 0])
        self.history['X_pred'] = X_pred

        if len(self.history['error'])> 1 and (error < self.history['error'][-1]):
            for pupil in self.pupils:
                pupil.update_local_loss()

        if self.fitness_func:
            train_mask = (F.ravel() < self.fitness_threshold) & (AE[:, 0] > self.tol)
        else:
            F = np.zeros_like(X_pred)
            train_mask = AE[:, 0] > self.tol

        return X_pred, error > self.tol, train_mask, F

    def _lesson_generator(self, training_content):
        condition = True
        while condition and (len(self.history['error']) < self.max_iter):

            prediction = np.array([pupil.predict(training_content) for pupil in self.pupils]).T
            X, condition, teach_idx, F = self._evaluate_performance(prediction, training_content)

            X = X[teach_idx, :]
            F = F[teach_idx, :]
            y = prediction[teach_idx, :]

            if self.guidance and (len(self.history['error']) > 1):
                X_pred = self.history['X_pred']
                distance = np.abs(training_content - X_pred)[teach_idx, 0]
                mask = np.array([d < np.deg2rad(200) for d in distance], dtype=bool)

                cont_filtered = training_content[teach_idx, :][mask, :]

                if cont_filtered.shape[0] > 0:
                    prediction = np.array([pupil.predict(cont_filtered) for pupil in self.pupils]).T
                    y = prediction
                    X = np.zeros((len(y), X_pred.shape[1]))
                    self._sim(y, X)

            if self.shuffle and self.lesson_size > 1:
                rng_state = np.random.get_state()
                np.random.shuffle(X)
                np.random.set_state(rng_state)
                np.random.shuffle(y)

            if self.lesson_size != np.inf:
                X = X[0:self.lesson_size, :]
                F = F[0:self.lesson_size, :]
                y = y[0:self.lesson_size, :]

            if self.plotter:
                plot_data = {}
                plot_data.update({'max_ae': np.rad2deg(np.max(self.history['activity'][-1]))})
                plot_data.update({'mean_ae': np.rad2deg(np.mean(self.history['activity'][-1]))})
                plot_data.update({'samples': [X[:, 0].tolist(), X[:, 2].tolist()]})
                self.plotter.update(plot_data)

            yield Lesson(X, F, y)

    def teach(self, training_content):
        try:
            for lesson in self._lesson_generator(training_content):
                for i, pupil in enumerate(self.pupils):
                    sucess = []
                    for ii, x in enumerate(lesson.X):
                        sucess.append(pupil.online_training(np.reshape(x, (1, -1)), lesson.y[ii, i].ravel(),
                                                            F=None, **self.kwargs))

                self.history['model_complexity'].append([pupil.M for pupil in self.pupils])

                print(f"[INFO]: Learned from {self.pupils[0].X.shape[0]} / {sum(sucess)}")

                if (len(self.history['model_complexity']) > 1) and\
                        not self.history['model_complexity'][-1] == self.history['model_complexity'][-2]:
                    print(f"[INFO]: Complexity of pupils after iteration {len(self.history['error'])-1}:"
                          f" {self.history['model_complexity'][-1]}")
        except KeyboardInterrupt:
            print("[WARNING]: User stop.")

        print(f"[INFO]: Training finished after {len(self.history['error'])-1} iterations "
              f"with max error of {np.rad2deg(self.history['error'][-1]):.3f}° "
              f"by means of {self.pupils[0].X.shape[0]} samples")

        return self.history







