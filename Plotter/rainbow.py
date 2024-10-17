import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

# https://stackoverflow.com/questions/31908982/multi-color-legend-entry

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/3 / len(orig_handle.colors) * i + width/3,
                                          -handlebox.ydescent],
                                         width/3 / len(orig_handle.colors),
                                         height,
                                         facecolor=c,
                                         edgecolor='none'))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch

if __name__ == "__main__":
    # ------ choose some colors
    colors1 = ['g', 'b', 'c', 'm', 'y']
    colors2 = ['k', 'r', 'k', 'r', 'k', 'r']

    # ------ create a dummy-plot (just to show that it works)
    f, ax = plt.subplots()
    ax.plot([1, 2, 3, 4, 5], [1, 4.5, 2, 5.5, 3], c='g', lw=0.5, ls='--',
            label='... just a line')
    ax.scatter(range(len(colors1)), range(len(colors1)), c=colors1)
    ax.scatter([range(len(colors2))], [.5] * len(colors2), c=colors2, s=50)

    # ------ get the legend-entries that are already attached to the axis
    h, l = ax.get_legend_handles_labels()

    # ------ append the multicolor legend patches
    h.append(MulticolorPatch(colors1))
    l.append("a nice multicolor legend patch")

    h.append(MulticolorPatch(colors2))
    l.append("and another one")

    # ------ create the legend
    f.legend(h, l, loc='upper left',
             handler_map={MulticolorPatch: MulticolorPatchHandler(),
                          MulticolorX: MulticolorXHandler()},
             bbox_to_anchor=(.125, .875))
    plt.show()