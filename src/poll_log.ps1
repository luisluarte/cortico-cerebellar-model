
$log_file = "C:\Users\DCCS5\Documents\GitHub\cortico-cerebellar-model\results\poll_remote.txt"
while ($true) {
    $out = ssh niconicoluarte@100.121.224.32 "tail -n 20 ~/cortico-cerebellar-model/src/r/struct_nsga2_fixed.log"
    if ($out -match "Cortico-Cerebellar Done") {
        Write-Host "FINISHED!"
        break
    }
    Write-Host "Still running..."
    Start-Sleep -Seconds 30
}

