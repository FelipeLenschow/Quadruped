// Browser-local UI state for the dashboard.
//
// One page holds both halves now -- the training curves and, below them, the evaluated
// checkpoints of the same runs -- so this is no longer about carrying a selection between
// documents; it is what survives a reload. Runs are keyed by their directory path under the repo
// (the tfevents sit in <run>, the eval reports in <run>/checkpoints/), which is what lets the
// checkpoint buttons attach to the run selected in the sidebar.
(function (global) {
    const KEY = 'quadrupedViewer.v1';

    function read() {
        try {
            return JSON.parse(localStorage.getItem(KEY)) || {};
        } catch (err) {
            return {};   // private browsing or a corrupted value: fall back to defaults
        }
    }

    let state = read();

    function write() {
        try {
            localStorage.setItem(KEY, JSON.stringify(state));
        } catch (err) {
            /* storage full or blocked -- the dashboard still works, it just forgets */
        }
    }

    global.ViewerState = {
        // Stored settings for one part of the page, e.g. ViewerState.get('training').
        get(section) {
            return state[section] || {};
        },
        // Merge and persist; only the keys passed in are touched.
        patch(section, values) {
            state[section] = Object.assign({}, state[section], values);
            write();
        },
    };
})(window);
