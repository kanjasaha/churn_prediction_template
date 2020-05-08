
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

class evaluate_metrics(object):
    
    def __init__(self, results):
        self.results=results
        


    def compare_classification_metrics(self):
        """
        Visualization code to display results of various learners.
        
        inputs:
          - learners: a list of supervised learners
          - stats: a list of dictionaries of the statistic results from 'train_predict()'
          
        """
      
        # Create figure
        fig, ax = plt.subplots(2, 3, figsize = (20,15))

        # Constants
        bar_width = 0.2
        colors = ['#A00000','#00A0A0','#00A000','#1b00a0']
        
        # Super loop to plot four panels of data
        for k, learner in enumerate(self.results.keys()):
            for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
                for i in np.arange(3):
                    
                    # Creative plot code
                    ax[j//3, j%3].bar(i+k*bar_width, self.results[learner][i][metric], width = bar_width, color = colors[k])
                    ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                    ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                    ax[j//3, j%3].set_xlabel("Set Size")
                    ax[j//3, j%3].set_xlim((-0.1, 3.0))
        
        # Add unique y-labels
        ax[0, 0].set_ylabel("Time (in seconds)")
        ax[0, 1].set_ylabel("Accuracy Score")
        ax[0, 2].set_ylabel("F-score")
        ax[1, 0].set_ylabel("Time (in seconds)")
        ax[1, 1].set_ylabel("Accuracy Score")
        ax[1, 2].set_ylabel("F-score")
        
        # Add titles
        ax[0, 0].set_title("Model Training")
        ax[0, 1].set_title("Accuracy Score on Training Set")
        ax[0, 2].set_title("F-score on Training Set")
        ax[1, 0].set_title("Model Predicting")
        ax[1, 1].set_title("Accuracy Score on Testing Set")
        ax[1, 2].set_title("F-score on Testing Set")
        
          
        # Set y-limits for score panels
        ax[0, 1].set_ylim((0, 1))
        ax[0, 2].set_ylim((0, 1))
        ax[1, 1].set_ylim((0, 1))
        ax[1, 2].set_ylim((0, 1))

        # Create patches for the legend
        patches = []
        for i, learner in enumerate(self.results.keys()):
            patches.append(mpatches.Patch(color = colors[i], label = learner))
        plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
                  loc = 'upper center', borderaxespad = 0., ncol = 4, fontsize = 'x-large')
        
        # Aesthetics
        plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
        plt.tight_layout()
        plt.show()
        