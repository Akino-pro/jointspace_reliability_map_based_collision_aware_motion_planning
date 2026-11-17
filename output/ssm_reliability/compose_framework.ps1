Add-Type -AssemblyName System.Drawing
$baseDir=Join-Path (Get-Location) 'output'
$assetDir=Join-Path $baseDir 'ssm_reliability'
$source=Join-Path $baseDir 'f275028a-51e6-4203-8ded-e665c70ce769.png'
$original=[System.Drawing.Bitmap]::FromFile($source)
$canvas=[System.Drawing.Bitmap]::new($original.Width,$original.Height)
$g=[System.Drawing.Graphics]::FromImage($canvas)
$g.DrawImageUnscaled($original,0,0)
$g.InterpolationMode='HighQualityBicubic'
$svg=[System.Text.StringBuilder]::new()
[void]$svg.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='1672' height='941' viewBox='0 0 1672 941'>")
$b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes($source))
[void]$svg.AppendLine("<image width='1672' height='941' href='data:image/png;base64,$b64'/>")
$placements=@(@(1,322,322,231,191),@(3,880,322,228,191),@(4,1142,322,223,191))
foreach($placement in $placements){
  $id,$x,$y,$w,$h=$placement
  $path=Join-Path $assetDir "framework_panel_$id.png"
  $img=[System.Drawing.Image]::FromFile($path)
  $g.FillRectangle([System.Drawing.Brushes]::White,$x,$y,$w,$h)
  $g.DrawImage($img,[System.Drawing.Rectangle]::new($x,$y,$w,$h))
  $b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes($path))
  [void]$svg.AppendLine("<rect x='$x' y='$y' width='$w' height='$h' fill='white'/><image x='$x' y='$y' width='$w' height='$h' preserveAspectRatio='none' href='data:image/png;base64,$b64'/>")
  $img.Dispose()
}
# The complete second card, and all pixels outside the three replacement rectangles,
# remain identical to the supplied original.
$same=$true
for($y=250;$y -lt 518;$y++){for($x=580;$x -lt 851;$x++){
  if($canvas.GetPixel($x,$y).ToArgb() -ne $original.GetPixel($x,$y).ToArgb()){$same=$false;break}
}}
if(-not $same){throw 'The retained second panel changed.'}
$canvas.Save((Join-Path $assetDir 'framework_3r_experiment.png'),[System.Drawing.Imaging.ImageFormat]::Png)
[void]$svg.AppendLine('</svg>')
[IO.File]::WriteAllText((Join-Path $assetDir 'framework_3r_experiment.svg'),$svg.ToString())
$g.Dispose();$canvas.Dispose();$original.Dispose()
Write-Output 'Saved exact data composite. Second panel verified pixel-identical.'
