// Browser-local UI state shared by the two viewer pages.
//
// The dashboard is two documents -- the evaluation hub (/) and the training curves
// (/training.html) -- so every hop between them is a full page load that would otherwise reset
// each selection and toggle. Both pages key runs the same way (a run id is its directory path
// under the repo: the eval reports sit in <run>/checkpoints/, the tfevents in <run> itself),
// which also lets the focused runs travel from one page to the other.
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
        // Stored settings for one page, e.g. ViewerState.get('eval').
        get(section) {
            return state[section] || {};
        },
        // Merge and persist; only the keys passed in are touched.
        patch(section, values) {
            state[section] = Object.assign({}, state[section], values);
            write();
        },
        // The runs the user last selected, on whichever page. Read by the other page to open on
        // the same runs -- only ever as a default, never overriding an explicit selection.
        focusedRuns() {
            return state.focusedRuns || [];
        },
        setFocusedRuns(ids) {
            state.focusedRuns = Array.from(new Set(ids));
            write();
        },
    };
})(window);
