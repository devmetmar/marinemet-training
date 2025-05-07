import numpy as np
import datetime
import os

class plotter:
    global MSG_INFO, MSG_WARN, MSG_ERR
    MSG_INFO = 1
    MSG_WARN = 2
    MSG_ERR = 3

    def __init__(self, img_dir=None):
        try:
            self.img_dir = img_dir
            os.makedirs(self.img_dir, exist_ok=True)
        except Exception as e:
            self.logging(self.MSG_ERR, f"Initialization failed: {e}")
            raise

    def logging(self, messType, messText):
        dtNow = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S :")
        code = {MSG_INFO: "I", MSG_WARN: "W", MSG_ERR: "E"}.get(messType, "E")
        print(f"{dtNow}\t{code}\t{messText}")
        
    def plot_aws_location(self, extents=None, points=None, awslocs1=None, awslocs2=None, filename=None, gridlines=False):
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if extents is None:
            extents = [90, 145, -15, 15]
            self.logging(MSG_INFO, f"Using default extents = {extents}")
        else:    
            self.logging(MSG_INFO, f"Using extents = {extents}")
        
        self.logging(MSG_INFO, "Creating plot, please kindly wait...")

        proj = ccrs.PlateCarree()
        cm = 1 / 2.54
        X = extents[1] - extents[0]
        Y = extents[3] - extents[2]
        rat = X / Y
        lth = 25

        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(lth * cm, lth * cm / rat), subplot_kw=dict(projection=proj))
        ax.set_extent(extents, crs=proj)
        ax.add_feature(cfeature.GSHHSFeature(scale="high", levels=[1, 2, 3, 4], facecolor="linen"), linewidth=.3)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        if gridlines:
            gl = ax.gridlines(crs=proj, draw_labels=True, dms=False, color='gray', alpha=.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
        if points is None:
            points = []
        if awslocs1 is None:
            awslocs1 = []
        if awslocs2 is None:
            awslocs2 = []

        if len(points) == 0 and len(awslocs1) == 0 and len(awslocs2) == 0:
            self.logging(MSG_INFO, "No data to plot, creating empty map.")
        else:
            if len(points) > 0:
                ax.plot(points[:,0], points[:,1], color='red', marker='o',
                       transform=proj, markersize=5, alpha=0.9, linestyle='None', markeredgecolor="None", label="Lokasi Data")
            if len(awslocs1) > 0:
                ax.plot(awslocs1[:, 0], awslocs1[:, 1], color='orange', marker='o',
                        transform=proj, markersize=5, alpha=0.9, linestyle='None', markeredgecolor="None", label="MAWS Existing")
            if len(awslocs2) > 0:
                ax.plot(awslocs2[:, 0], awslocs2[:, 1], color='green', marker='o',
                        transform=proj, markersize=5, alpha=0.9, linestyle='None', markeredgecolor="None", label="MAWS MMS1")

            ax.legend(loc='lower left')

        if filename:
            filepath = os.path.join(self.img_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logging(MSG_INFO, f"Figure saved to {filepath}")
        else:
            plt.show()