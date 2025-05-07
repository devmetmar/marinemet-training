import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.grid_finder as gf
import mpl_toolkits.axisartist.floating_axes as fa
import warnings
warnings.filterwarnings('ignore')



def verification_metrics(obs_df, model_df):
    # Pastikan waktu dan kedalaman cocok
    obs_df, model_df = obs_df.align(model_df, join='inner', axis=1)
    obs_df, model_df = obs_df.align(model_df, join='inner', axis=0)

    # Flatten data agar bisa dihitung metrik
    obs = obs_df.values.flatten()
    mod = model_df.values.flatten()

    # Mask NaN
    valid_mask = ~np.isnan(obs) & ~np.isnan(mod)
    obs_valid = obs[valid_mask]
    mod_valid = mod[valid_mask]

    # Hitung metrik
    correlation = np.corrcoef(obs_valid, mod_valid)[0, 1]
    mape = np.mean(np.abs((obs_valid - mod_valid) / obs_valid)) * 100
    bias = np.mean(mod_valid - obs_valid)
    rmse = np.sqrt(np.mean((mod_valid - obs_valid)**2))
    std_obs = np.std(obs_valid)
    std_mod = np.std(mod_valid)

    return {
        'Correlation': correlation,
        'MAPE (%)': mape,
        'Bias': bias,
        'RMSE': rmse,
        'Std Obs': std_obs,
        'Std Model': std_mod
    }


class TaylorDiagram(object):
  def __init__(self, STD ,fig=None, rect=111, label='_',grid=False):
    self.STD = STD
    tr = PolarAxes.PolarTransform()
    # Correlation labels
    rlocs = np.concatenate(((np.arange(11.0) / 10.0), [0.95, 0.99]))
    tlocs = np.arccos(rlocs) # Conversion to polar angles
    gl1 = gf.FixedLocator(tlocs) # Positions
    tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
    # Standard deviation axis extent
    self.smin = 0
    self.smax = 15
    gh = fa.GridHelperCurveLinear(tr,extremes=(0,(np.pi/2),self.smin,self.smax),grid_locator1=gl1,tick_formatter1=tf1,)
    if fig is None:
      fig = plt.figure()
    ax = fa.FloatingSubplot(fig, rect, grid_helper=gh)
    fig.add_subplot(ax)
    # Angle axis
    ax.axis['top'].set_axis_direction('bottom')
    ax.axis['top'].label.set_text("Correlation coefficient")
    ax.axis['top'].toggle(ticklabels=True, label=True)
    ax.axis['top'].major_ticklabels.set_axis_direction('top')
    ax.axis['top'].label.set_axis_direction('top')
    # X axis
    ax.axis['left'].set_axis_direction('bottom')
    ax.axis['left'].label.set_text("Standard deviation")
    ax.axis['left'].toggle(ticklabels=True, label=True)
    ax.axis['left'].major_ticklabels.set_axis_direction('bottom')
    ax.axis['left'].label.set_axis_direction('bottom')
    # Y axis
    ax.axis['right'].set_axis_direction('top')
    ax.axis['right'].label.set_text("Standard deviation")
    ax.axis['right'].toggle(ticklabels=True, label=True)
    ax.axis['right'].major_ticklabels.set_axis_direction('left')
    ax.axis['right'].label.set_axis_direction('top')
    # Useless
    ax.axis['bottom'].set_visible(False)
    ax.set_facecolor('none')
    # Contours along standard deviations
    if grid==True:
      ax.grid()
    self._ax = ax # Graphical axes
    self.ax = ax.get_aux_axes(tr) # Polar coordinates
    # Add reference point and STD contour
    l , = self.ax.plot([0], self.STD, '', ls='', ms=7, label=label)
    l1 , = self.ax.plot([0], self.STD, '', ls='', ms=7, label=label)
    t = np.linspace(0, (np.pi / 2.0))
    t1 = np.linspace(0, (np.pi / 2.0))
    r = np.zeros_like(t) + self.STD
    r1 = np.zeros_like(t) + self.STD
    ref=self.ax.plot(t, r, 'r--',lw=.5,label='_')
    # Collect sample points for latter use (e.g. legend)
    self.samplePoints = [l]
    self.samplePoints = [l1]
    
  def add_sample(self,STD,r,*args,**kwargs):
    l,= self.ax.plot(np.arccos(r), STD, *args, **kwargs) # (theta, radius)
    self.samplePoints.append(l)
    return l

  def add_sample(self,STD,r1,*args,**kwargs):
    l1,= self.ax.plot(np.arccos(r1), STD, *args, **kwargs) # (theta, radius)
    self.samplePoints.append(l1)
    return l1

  def add_contours(self,levels=10,**kwargs):
    rs, ts = np.meshgrid(np.linspace(self.smin, self.smax), np.linspace(0, (np.pi / 2.0)))
    RMSE=np.sqrt(np.power(self.STD, 2) + np.power(rs, 2) - (2.0 * self.STD * rs  *np.cos(ts)))
    contours = self.ax.contour(ts, rs, RMSE, levels, alpha=.8,linestyles='dashdot', **kwargs)
    return contours