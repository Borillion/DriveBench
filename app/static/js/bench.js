 $(function() {
      $('button#fire').bind('click', function() {
        $.getJSON('/echo', {
          engine : $('input[name="engine"]:checked').val(),
          drive : $('input[name="drive"]:checked').val(),
          pattern : $('input[name="pattern"]:checked').val(),
          instances : $('input[name="instances"]:checked').val()
        }, function(data) {
          $("#result").replaceWith('<div id="result"><img src=' + data.response + '></div>');
        });
        return false;
      });
    });
