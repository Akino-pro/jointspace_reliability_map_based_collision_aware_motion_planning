Add-Type -AssemblyName System.Drawing
$outDir = $PSScriptRoot
if (-not $outDir) { $outDir = Join-Path (Get-Location) 'output/project_framework' }
$bmp = New-Object System.Drawing.Bitmap(3840,2160)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.ScaleTransform(2,2)
$g.SmoothingMode = 'AntiAlias'
$g.TextRenderingHint = 'AntiAliasGridFit'
$g.Clear([System.Drawing.Color]::White)
$svg = [System.Text.StringBuilder]::new()
[void]$svg.AppendLine('<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080"><rect width="1920" height="1080" fill="white"/>')
function Col($c) { [System.Drawing.ColorTranslator]::FromHtml($c) }
function Box($x,$y,$w,$h,$fill,$stroke) {
  $b = [System.Drawing.SolidBrush]::new((Col $fill)); $p = [System.Drawing.Pen]::new((Col $stroke),1.5)
  $g.FillRectangle($b,$x,$y,$w,$h); $g.DrawRectangle($p,$x,$y,$w,$h)
  [void]$svg.AppendLine("<rect x='$x' y='$y' width='$w' height='$h' fill='$fill' stroke='$stroke' stroke-width='1.5'/>")
  $b.Dispose(); $p.Dispose()
}
function Txt($x,$y,$w,$h,$s,$size=26,$color='#172B46',$bold=$false,$align='Center') {
  $style = [System.Drawing.FontStyle]::Regular; if($bold){$style=[System.Drawing.FontStyle]::Bold}
  $font=[System.Drawing.Font]::new('Arial',[single]$size,$style,[System.Drawing.GraphicsUnit]::Pixel)
  $brush=[System.Drawing.SolidBrush]::new((Col $color)); $fmt=[System.Drawing.StringFormat]::new()
  $fmt.Alignment=[System.Drawing.StringAlignment]::$align; $fmt.LineAlignment='Center'
  $g.DrawString($s,$font,$brush,[System.Drawing.RectangleF]::new($x,$y,$w,$h),$fmt)
  $lines=$s.Split("`n"); $lh=$size*1.16; $start=$y+$h/2-($lines.Count-1)*$lh/2
  $anchor='middle'; $tx=$x+$w/2; if($align -eq 'Near'){$anchor='start';$tx=$x}
  $weight='400'; if($bold){$weight='700'}
  for($i=0;$i -lt $lines.Count;$i++){
    $safe=[System.Security.SecurityElement]::Escape($lines[$i]); $ty=$start+$i*$lh
    [void]$svg.AppendLine("<text x='$tx' y='$ty' dominant-baseline='central' text-anchor='$anchor' font-family='Arial, sans-serif' font-size='$size' font-weight='$weight' fill='$color'>$safe</text>")
  }
  $font.Dispose();$brush.Dispose();$fmt.Dispose()
}
function Arrow($pts,$color='#60758C',$dash=$false) {
  $p=[System.Drawing.Pen]::new((Col $color),2.5)
  if($dash){$p.DashStyle='Dash'}
  $points=@(); foreach($xy in $pts){$points+= [System.Drawing.PointF]::new($xy[0],$xy[1])}
  $g.DrawLines($p,[System.Drawing.PointF[]]$points)
  $a=$pts[-2];$z=$pts[-1];$angle=[Math]::Atan2($z[1]-$a[1],$z[0]-$a[0]);$sz=11
  $tip=@([System.Drawing.PointF]::new($z[0],$z[1]),[System.Drawing.PointF]::new($z[0]-$sz*[Math]::Cos($angle-0.45),$z[1]-$sz*[Math]::Sin($angle-0.45)),[System.Drawing.PointF]::new($z[0]-$sz*[Math]::Cos($angle+0.45),$z[1]-$sz*[Math]::Sin($angle+0.45)))
  $b=[System.Drawing.SolidBrush]::new((Col $color));$g.FillPolygon($b,[System.Drawing.PointF[]]$tip)
  $coords=($pts|ForEach-Object {"$($_[0]),$($_[1])"}) -join ' '
  $d='';if($dash){$d="stroke-dasharray='8 6'"}
  [void]$svg.AppendLine("<polyline points='$coords' fill='none' stroke='$color' stroke-width='2.5' $d/>")
  $tri=($tip|ForEach-Object {"$($_.X),$($_.Y)"}) -join ' '
  [void]$svg.AppendLine("<polygon points='$tri' fill='$color'/>")
  $b.Dispose();$p.Dispose()
}
Txt 60 45 1800 72 'Reliability-Aware Motion Planning Framework' 44 '#172B46' $true 'Near'
Box 385 195 1130 360 '#F2F6FC' '#CBD9EC'
Box 385 635 1130 275 '#F0F8F6' '#C5E0D8'
Txt 415 211 1060 64 'Pre-failure motion planning' 32 '#28578A' $true 'Near'
Txt 415 651 1060 64 'Post-failure replanning' 32 '#18745E' $true 'Near'
Box 40 310 270 380 '#F7F9FC' '#D3DCE7'
Txt 60 328 230 52 'INPUTS' 25 '#60758C' $true
Txt 60 397 230 50 'Task' 27
Txt 60 464 230 50 'Environment' 27
Txt 55 529 240 50 'Robot parameters' 26
Txt 55 590 240 70 "Joint failure`nprobabilities" 26
Arrow @(@(310,500),@(345,500),@(345,370),@(420,370))
Arrow @(@(345,500),@(345,795),@(420,795))
$xs=@(420,695,970,1245)
$labels=@("Compute`ncollision-free region","Compute joint-space`nreliability map","Intersect region`nand reliability map`nto form graph","Optimize`ntrajectory")
for($i=0;$i -lt 4;$i++){
  Box $xs[$i] 310 235 145 '#FFFFFF' '#9CB7D7'
  Txt ($xs[$i]+12) 315 211 35 ('0'+($i+1)) 19 '#4975A3' $true 'Near'
  $textSize=24; if($i -eq 2){$textSize=22}
  Txt ($xs[$i]+8) 351 219 92 $labels[$i] $textSize '#172B46' $true
}
Arrow @(@(655,380),@(695,380)) '#4975A3'
Arrow @(@(930,380),@(970,380)) '#4975A3'
Arrow @(@(1205,380),@(1245,380)) '#4975A3'
Arrow @(@(537,455),@(537,515),@(1087,515),@(1087,455)) '#4975A3'
Txt 630 516 370 35 'Collision-free region' 20 '#4975A3'
Box 420 740 235 115 '#FFFFFF' '#8ABDAC'
Txt 428 754 219 87 "Joint failure`noccurs" 26 '#18694F' $true
Box 805 740 295 115 '#FFFFFF' '#8ABDAC'
Txt 815 750 275 94 "Use collision-free`nregion as graph" 25 '#18694F' $true
Box 1245 740 235 115 '#FFFFFF' '#8ABDAC'
Txt 1255 754 215 87 "Optimize`ntrajectory" 26 '#18694F' $true
Arrow @(@(655,795),@(805,795)) '#27866D'
Arrow @(@(1100,795),@(1245,795)) '#27866D'
Arrow @(@(537,515),@(537,595),@(952,595),@(952,740)) '#27866D' $true
Box 671 575 300 40 '#FFFFFF' '#FFFFFF'
Txt 675 575 292 40 'Reuse collision-free region' 21 '#18745E' $true
Box 1595 410 285 290 '#EAF2FB' '#90ABCA'
Txt 1610 427 255 45 'OUTPUT' 24 '#4975A3' $true
Txt 1605 485 265 140 "Collision-free`ntrajectory with`nhigh reliability" 29 '#172B46' $true
Arrow @(@(1480,380),@(1545,380),@(1545,500),@(1595,500)) '#4975A3'
Arrow @(@(1480,795),@(1545,795),@(1545,610),@(1595,610)) '#27866D'
[void]$svg.AppendLine('</svg>')
[System.IO.File]::WriteAllText((Join-Path $outDir 'project_framework.svg'),$svg.ToString(),[System.Text.UTF8Encoding]::new($false))
$bmp.Save((Join-Path $outDir 'project_framework.png'),[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose();$bmp.Dispose()
Get-Item (Join-Path $outDir 'project_framework.svg'),(Join-Path $outDir 'project_framework.png') | Select-Object FullName,Length
