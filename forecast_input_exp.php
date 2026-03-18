<?php
  header('content-type: application/json');
  if (array_key_exists('exp',$_REQUEST)){
    if (array_key_exists('model_type', $_REQUEST)) {
      $j = file_get_contents('forecast_input_' . $_REQUEST['exp'] . '_' . $_REQUEST['model_type'] . '.json');
    } else {
    $j = file_get_contents('forecast_input_' . $_REQUEST['exp'] . '.json');
    }
  } else {
    $j = file_get_contents('forecast_input.json');
  }
  $jj = json_decode($j,true);
  if (array_key_exists('l0',$_REQUEST)){
    $jj = $jj[$_REQUEST['l0']];
  }
  if (array_key_exists('l1',$_REQUEST)){
    $jj = $jj[$_REQUEST['l1']];
  }
  if (array_key_exists('l2',$_REQUEST)){
    $jj = $jj[$_REQUEST['l2']];
  }
  if (array_key_exists('l3',$_REQUEST)){
    $jj = $jj[$_REQUEST['l3']];
  }
  if (array_key_exists('l4',$_REQUEST)){
    $jj = $jj[$_REQUEST['l4']];
  }
  echo json_encode($jj,JSON_PRETTY_PRINT);
?>
