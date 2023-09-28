const csrftoken = Cookies.get('csrftoken');
$.ajaxSetup({
  beforeSend: function(xhr, settings) {
    xhr.setRequestHeader('X-CSRFToken', csrftoken);
  }
});

var editor = CodeMirror.fromTextArea(document.getElementById('text'), {
  lineNumbers: true,
});

$(document).ready(function() {
  editor.setSize($('.wrapper6').width(), $('.wrapper6').height());
});

window.onresize = function() {
  editor.setSize($('.wrapper6').width(), $('.wrapper6').height());
};

// ask for notifications permission
if (Notification.permission !== 'granted') {
  Notification.requestPermission();
}

function goto_line(line) {
  // var lines = $('.text').val().split('\n');
  // var height = $('.text').height();
  // var line_height = height / lines.length;
  // var top = line * line_height;
  // $('.text').scrollTop(top);

  editor.setCursor(line - 1);
  editor.focus();
}

function parse_asm(asm) {
  var height = $('.code').height();

  var re_header = /;-+$/gm;
  asm = asm.split('\n');
  new_asm = '';
  in_section = false;
  for (var i = 0; i < asm.length;) {
    if (!in_section) {
      in_section = true;
      new_asm += '<span class="section"><span class="header">\n'
      while (i < asm.length && asm[i].startsWith(';')) {
        new_asm += asm[i] + '\n';
        i++;
      }
      new_asm += '</span>\n';
    } else {
      in_section = false;
      new_asm += '<span class="content">\n';
      while (i < asm.length && !re_header.test(asm[i])) {
        new_asm += asm[i] + '\n';
        i++;
      }
      new_asm += '</span>\n';
    }
    if (i == asm.length) new_asm += '</span>\n';
  }
  asm = new_asm;

  var re_comment = /;.*\n/g;
  asm = asm.replace(re_comment, function(match) {
    return '<span class="comment">' + match.replace('\n', '') + '</span>\n';
  });

  var re_label = /^.+:$/gm;
  asm = asm.replace(re_label, function(match) {
    return '<span class="label">' + match + '</span>';
  });

  var re_rel = /\[rel \w+\]/g;
  asm = asm.replace(re_rel, function(match) {
    return '<span class="rel">' + match + '</span>';
  });

  var re_instr = /\t\w+\t/g;
  asm = asm.replace(re_instr, function(match) {
    return '<span class="instr">' + match + '</span>';
  });

  var re_num = /[\s,](\d+|#?0x([0-9a-f]+))\s/g;
  asm = asm.replace(re_num, function(match) {
    return match[0] + '<span class="number">' + match.slice(1) + '</span>';
  });

  var re_memory = /\(byte|[ddq]?word\)/g;
  asm = asm.replace(re_memory, function(match) {
    return '<span class="memory">' + match + '</span>';
  });

  var re_string = /'[^']*'/g;
  asm = asm.replace(re_string, function(match) {
    return '<span class="string">' + match + '</span>';
  });

  var re_link = /temp\.c:[0-9]+/g;
  asm = asm.replace(re_link, function(match) {
    var line = match.split(':')[1];
    return '<a href="#" onclick="goto_line(' + line + ')">' + match + '</a>';
  });

  asm = asm.replace(/\n<span class="section">/g, '<span class="section">');
  asm = asm.replace(/<span class="section">\n/g, '<span class="section">');
  asm = asm.replace(/\n<span class="header">/g, '<span class="header">');
  asm = asm.replace(/\n<span class="content">/g, '<span class="content">');
  asm = asm.replace(/<span class="content">\n/g, '<span class="content">');
  asm = asm.replace(/<span class="header">\n/g, '<span class="header">');

  $('.code').html('<pre>' + asm + '</pre>');
  $('.code').height(height);

  // add click event to header spans
  $('.header').click(function() {
    console.log('click');
    $(this).next().toggle();
  });
}

function parse_err(err) {
  var height = $('.code').height();

  $('.code').html('<pre>' + 'sdcc: <a href="#" onclick="goto_line('+ 36 + ')">' + 'test.c:36' + '</a>: syntax error: token -> \'}\' ; column 1</pre>');
  $('.code').height(height);
}

$(document).ready(function() {
  // load tree
  $.ajax({
    url: '/load_tree/',
    type: 'GET',
    success: function(data) {
      $('.tree').html(data.tree);
    }
  });

  // load text
  $.ajax({
    url: '/load_file/',
    type: 'GET',
    success: function(data) {
      // $('.text').val(data.content);
      editor.setValue(data.content);
    }
  });
});

function add_child(parent_id, name, type) {
  var data = {
    'parent_id': parent_id,
    'name': name,
    'type': type
  };

  $.ajax({
    url: '/add_child/',
    type: 'POST',
    data: data,
    success: function(data) {
      $('.tree').html(data.tree);
    }
  });
}

$(document).on('click', '.add-file-btn', function() {
  var parent_id = $(this).parent().data('id');
  var name = prompt('Enter file name:');
  if (name) add_child(parent_id, name, 'file');
});

$(document).on('click', '.add-dir-btn', function() {
  var parent_id = $(this).parent().data('id');
  var name = prompt('Enter directory name:');
  if (name) add_child(parent_id, name, 'directory');
});

$(document).on('click', '.delete-btn', function() {
  var parent_id = $(this).parent().data('id');
  var type = $(this).parent().attr('class') == 'folder' ? 'directory' : 'file';
  var data = { 'item_id': parent_id, 'type': type };

  $.ajax({
    url: '/delete_item/',
    type: 'POST',
    data: data,
    success: function(data) {
      $('.tree').html(data.tree);
    }
  });
});

$(document).on('click', '.file-select-btn', function() {
  var file_id = $(this).parent().data('id');
  var data = { 'file_id': file_id };

  $.ajax({
    url: '/select_file/',
    type: 'POST',
    data: data,
    success: function(data) {
      if (data.success == false) {
        if (Notification.permission === 'granted') {
          var options = {
            body: 'Error selecting file!',
            priority: -1
          };
          var n = new Notification('Notification', options);
        } else {
          alert('Error selecting file!');
        }
        return;
      }
      // $('.text').val(data.content);
      editor.setValue(data.content);
    },
    error: function(data) {
      if (Notification.permission === 'granted') {
        var options = {
          body: 'Error selecting file!',
          priority: -1
        };
        var n = new Notification('Notification', options);
      } else {
        alert('Error selecting file!');
      }
    }
  });
});

$(document).on('click', '.save-file-btn', function() {
  // var data = { 'content': $('.text').val() };
  var data = { 'content': editor.getValue() };

  $.ajax({
    url: '/save_file/',
    type: 'POST',
    data: data,
    success: function(data) {
      if (data.success == false) {
        if (Notification.permission === 'granted') {
          var options = {
            body: 'Error saving file!',
            priority: -1
          };
          var n = new Notification('Notification', options);
        } else {
          alert('Error saving file!');
        }
        return;
      }
      if (Notification.permission === 'granted') {
        var options = {
          body: 'File saved successfully!',
          priority: -1
        };
        var n = new Notification('Notification', options);
      }
    },
    error: function(data) {
      if (Notification.permission === 'granted') {
        var options = {
          body: 'Error saving file!',
          priority: -1
        };
        var n = new Notification('Notification', options);
      } else {
        alert('Error saving file!');
      }
    },
  });
});

// compiler options

function clear_cstd_buttons() {
  $('.c89').removeClass('active');
  $('.c99').removeClass('active');
  $('.c11').removeClass('active');
}

$(document).on('click', '.c89', function() {
  clear_cstd_buttons();
  $(this).addClass('active');
});

$(document).on('click', '.c99', function() {
  clear_cstd_buttons();
  $(this).addClass('active');
});

$(document).on('click', '.c11', function() {
  clear_cstd_buttons();
  $(this).addClass('active');
});

$(document).on('click', '.opt-code-speed', function() {
  $(this).toggleClass('active');
});

$(document).on('click', '.opt-code-size', function() {
  $(this).toggleClass('active');
});

$(document).on('click', '.peep-asm', function() {
  $(this).toggleClass('active');
});

const processor_dependent_options = [
  '#mmcs51-dep .d1-model-small',
  '#mmcs51-dep .d1-model-medium',
  '#mmcs51-dep .d1-model-large',
  '#mz80-dep .d2-fno-omit-frame-pointer',
  '#mz80-dep .d2-reserve-regs-iy',
  '#mstm8-dep .d3-model-medium',
  '#mstm8-dep .d3-model-large',
];

function clear_processor_buttons() {
  $('.mmcs51').removeClass('active');
  $('.mz80').removeClass('active');
  $('.mstm8').removeClass('active');
  $('#mmcs51-dep').hide();
  $('#mz80-dep').hide();
  $('#mstm8-dep').hide();
  for (opt of processor_dependent_options) $(opt).removeClass('active');
}

$(document).on('click', '.mmcs51', function() {
  clear_processor_buttons();
  $(this).addClass('active');
  $('#mmcs51-dep').show();
});

$(document).on('click', '.mz80', function() {
  clear_processor_buttons();
  $(this).addClass('active');
  $('#mz80-dep').show();
});

$(document).on('click', '.mstm8', function() {
  clear_processor_buttons();
  $(this).addClass('active');
  $('#mstm8-dep').show();
});

for (opt of processor_dependent_options) {
  $(document).on('click', opt, function() {
    $(this).toggleClass('active');
  });
}

clear_cstd_buttons();
clear_processor_buttons();
$('.c99').addClass('active');
$('.mmcs51').addClass('active');
$('#mmcs51-dep').show();

// compiler
$(document).on('click', '.compile-btn', function() {
  data = {};

  // get c standard
  var cstd = 'c89';
  if ($('.c99').hasClass('active')) cstd = 'c99';
  if ($('.c11').hasClass('active')) cstd = 'c11';
  data['cstd'] = '--std-' + cstd;

  // get optimizations
  var opt = '';
  if ($('.opt-code-speed').hasClass('active')) opt += '--opt-code-speed ';
  if ($('.opt-code-size').hasClass('active')) opt += '--opt-code-size ';
  if ($('.peep-asm').hasClass('active')) opt += '--peep-asm ';
  data['opt'] = opt.trim();

  var proc = '';
  if ($('.mmcs51').hasClass('active')) proc = '-mmcs51';
  if ($('.mz80').hasClass('active')) proc = '-mz80';
  if ($('.mstm8').hasClass('active')) proc = '-mstm8';
  data['proc'] = proc;

  var proc_opt = '';
  if ($('#mmcs51-dep .d1-model-small').hasClass('active')) proc_opt += '--model-small ';
  if ($('#mmcs51-dep .d1-model-medium').hasClass('active')) proc_opt += '--model-medium ';
  if ($('#mmcs51-dep .d1-model-large').hasClass('active')) proc_opt += '--model-large ';
  if ($('#mz80-dep .d2-fno-omit-frame-pointer').hasClass('active')) proc_opt += '--fno-omit-frame-pointer ';
  if ($('#mz80-dep .d2-reserve-regs-iy').hasClass('active')) proc_opt += '--reserve-regs-iy ';
  if ($('#mstm8-dep .d3-model-medium').hasClass('active')) proc_opt += '--model-medium ';
  if ($('#mstm8-dep .d3-model-large').hasClass('active')) proc_opt += '--model-large ';
  data['proc_opt'] = proc_opt.trim();

  $.ajax({
    url: '/compile/',
    type: 'GET',
    data: data,
    success: function(data) {
      if (data.success == false) {
        parse_err(data.stderr);
        return;
      }
      if (data.asm == '') {
        console.log('err');
        parse_err(data.stderr);
        if (data.stderr == '')
          parse_err('No output');
      } else {
        parse_asm(data.asm);
      }
    },
    error: function(data) {
      $('.code').val('Error running sdcc');
    }
  });
});

$(document).on('click', '.compile-and-download-btn', function() {
  data = {};

  // get c standard
  var cstd = 'c89';
  if ($('.c99').hasClass('active')) cstd = 'c99';
  if ($('.c11').hasClass('active')) cstd = 'c11';
  data['cstd'] = '--std-' + cstd;

  // get optimizations
  var opt = '';
  if ($('.opt-code-speed').hasClass('active')) opt += '--opt-code-speed ';
  if ($('.opt-code-size').hasClass('active')) opt += '--opt-code-size ';
  if ($('.peep-asm').hasClass('active')) opt += '--peep-asm ';
  data['opt'] = opt.trim();

  var proc = '';
  if ($('.mmcs51').hasClass('active')) proc = '-mmcs51';
  if ($('.mz80').hasClass('active')) proc = '-mz80';
  if ($('.mstm8').hasClass('active')) proc = '-mstm8';
  data['proc'] = proc;

  var proc_opt = '';
  if ($('#mmcs51-dep .d1-model-small').hasClass('active')) proc_opt += '--model-small ';
  if ($('#mmcs51-dep .d1-model-medium').hasClass('active')) proc_opt += '--model-medium ';
  if ($('#mmcs51-dep .d1-model-large').hasClass('active')) proc_opt += '--model-large ';
  if ($('#mz80-dep .d2-fno-omit-frame-pointer').hasClass('active')) proc_opt += '--fno-omit-frame-pointer ';
  if ($('#mz80-dep .d2-reserve-regs-iy').hasClass('active')) proc_opt += '--reserve-regs-iy ';
  if ($('#mstm8-dep .d3-model-medium').hasClass('active')) proc_opt += '--model-medium ';
  if ($('#mstm8-dep .d3-model-large').hasClass('active')) proc_opt += '--model-large ';
  data['proc_opt'] = proc_opt.trim();

  $.ajax({
    url: '/compile/',
    type: 'GET',
    data: data,
    success: function(data) {
      if (data.success == false) {
        if (Notification.permission == 'granted') {
          n = new Notification('Notification', "Download failed");
        } else {
          alert('Download failed');
        }
        return;
      }

      var blob = new Blob([data.asm]);
      var link = document.createElement('a');
      link.href = window.URL.createObjectURL(blob);
      link.download = data.filename;
      link.click();
      link.remove();
      window.URL.revokeObjectURL(link.href);
    },
    error: function(data) {
      if (Notification.permission == 'granted') {
        n = new Notification('Notification', "Download failed");
      } else {
        alert('Download failed');
      }
    }
  });
});

// drag and drop files
$(document).on('dragover', '.text', function(e) {
  e.preventDefault();
  e.stopPropagation();
  $(this).addClass('dragover');
});

$(document).on('dragleave', '.text', function(e) {
  e.preventDefault();
  e.stopPropagation();
  $(this).removeClass('dragover');
});

$(document).on('drop', '.text', function(e) {
  e.preventDefault();
  e.stopPropagation();
  $(this).removeClass('dragover');

  var files = e.originalEvent.dataTransfer.files;
  if (files.length == 0) return;
  var file = files[0];

  var reader = new FileReader();
  reader.onload = function(e) {
    content = e.target.result;
    // $('.text').val(content);
    editor.setValue(content);
  };

  reader.readAsText(file);
});
