/** Reloads all data
 * @param ft footable instance
 * @param btn jquery object with the clicked button (to disable/enable it while loading)
 * @param buttons jquery handle to the control buttons (to hide them )
 */
function reload_data(ft, btn) {
    return function() {
        var numrows = 0;
        var txt = btn.text();
        btn.prop('disabled', true).text('...');
        ft.$el.hide();   // Hide the header while we work
        // console.log("Reloading...");
        $.get('/api/rows').done(function(data) {
            ft.rows.load(data, false);  // don't append
            numrows = data.length;
            // console.log(numrows + " rows read.")
        }).then(function () {
            btn.prop('disabled', false).text(txt);
            ft.$el.show();
        });
    }
}

/** Creates the event handler for the Reload button
 *
 * @param ft A handle to the footable object
 * @returns {Function} The event handler itself. Takes the event as argument
 */
function handle_reload(ft) {
    return function(e) {
        e.preventDefault();
        $.get("/api/reload").done(reload_data(ft, $(e.currentTarget)));
    }
}


/** Creates the event handler for Delete button
 *
 * @param ft A handle to the footable object
 * @returns {Function} The event handler itself. Takes the event as argument
 */
function handle_delete(ft) {
    return function(e) {
        e.preventDefault();
        var active = $('.tgl:checked').map(function () { return this.id; }).get();
        $.alertable.confirm('Delete: <br/><small>' + active.join(',<br/>')+ '</small>', {html: true}).then(function() {
            $.get("/api/delete/" + active.join()).done(reload_data(ft, $(e.currentTarget)));
        }, function(e) {
            console.log('Deletion canceled');
        });
    }
}


/** Creates the event handler for Plot multiple button
 *
 * @param ft A handle to the footable object
 * @returns {Function} The event handler itself. Takes the event as argument
 */
function handle_plot_multiple(ft) {
    return function(e) {
        e.preventDefault();
        var active = $('.tgl:checked').map(function () { return this.id; }).get();
        $.get("/api/plot_multiple/" + active.join(), function(data) {
            $('#multiple_plot').attr('src', 'data:image/jpeg;base64, ' + data);
            var url = location.href;
            location.href = "#multiple_plot_target";
            history.replaceState(null, "", url);
        });
    }
}


/**
 * Change the src of the lightshow div where single plots are drawn, so that it
 * is activated (by the :target property in CSS) and it displays the image.
 * @param e
 */
function show_one(e) {
    $('#single_plot').attr('src', '/api/plot_one/' + e.getAttribute('data-value'));
}


/**
 * Change the src of the lightshow div where single plots are drawn, so that it
 * is activated (by the :target property in CSS) and it displays the image.
 * @param e
 */
function show_mesh(e) {
    $('#single_plot').attr('src', '/api/plot_mesh/' + e.getAttribute('data-value'));
}


/** Creates a function to move the additional buttons into footable's header
 *
 * @param btns Handle to the list of control buttons to insert.
 * @returns {Function} handling the event postdraw.ft.table.
 * FIXME: floatThead breaks sorting! We use stickyTableHeaders() instead, which is not as good.
 * See https://github.com/mkoryak/floatThead/issues/37
 * and https://github.com/mkoryak/bootstrap-sortable/commit/23004f565f80aed405120c34449b43c6e2ec1167
 */
function fix_header(btns) {
    return function(e, ft) {
        /*HACK! I need to delay moving the element or it will just disappear*/
        $(e.currentTarget.tHead).delay(200).queue(function () {
            btns.detach();
            ft.$el.find('.form-inline').prepend(btns);
            ft.$el.stickyTableHeaders();
            btns.css('display', 'inline-flex');
        });
    }
}
