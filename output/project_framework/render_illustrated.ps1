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
function Line($pts,$color='#4975A3',$width=2) {
  $p=[System.Drawing.Pen]::new((Col $color),$width)
  $arr=@(); foreach($xy in $pts){$arr += [System.Drawing.PointF]::new($xy[0],$xy[1])}
  $g.DrawLines($p,[System.Drawing.PointF[]]$arr);$p.Dispose()
  $coords=($pts|ForEach-Object {"$($_[0]),$($_[1])"}) -join ' '
  [void]$svg.AppendLine("<polyline points='$coords' fill='none' stroke='$color' stroke-width='$width' stroke-linecap='round' stroke-linejoin='round'/>")
}
function Dot($x,$y,$r,$fill,$stroke='#FFFFFF') {
  $b=[System.Drawing.SolidBrush]::new((Col $fill));$p=[System.Drawing.Pen]::new((Col $stroke),1.5)
  $g.FillEllipse($b,($x-$r),($y-$r),(2*$r),(2*$r));$g.DrawEllipse($p,($x-$r),($y-$r),(2*$r),(2*$r));$p.Dispose();$b.Dispose()
  [void]$svg.AppendLine("<circle cx='$x' cy='$y' r='$r' fill='$fill' stroke='$stroke' stroke-width='1.5'/>")
}
function Region($x,$y,$graph=$false,$trajectory=$false,$color='#4975A3') {
  Box $x $y 140 78 '#ECF3F9' '#CAD8E6'
  for($j=1;$j -lt 7;$j++){Line @(@(($x+$j*20),$y),@(($x+$j*20),($y+78))) '#DCE5EE' 1}
  for($j=1;$j -lt 4;$j++){Line @(@($x,($y+$j*20)),@(($x+140),($y+$j*20))) '#DCE5EE' 1}
  Box ($x+32) ($y+7) 33 30 '#CCD3DC' '#BCC6D2'
  Box ($x+81) ($y+46) 34 27 '#CCD3DC' '#BCC6D2'
  if($graph){
    $nodes=@(@(10,65),@(32,47),@(69,42),@(85,17),@(126,12),@(124,40),@(64,68),@(14,14))
    $edges=@(@(0,1),@(1,2),@(2,3),@(3,4),@(4,5),@(2,5),@(0,6),@(1,6),@(1,7))
    foreach($e in $edges){$a=$nodes[$e[0]];$z=$nodes[$e[1]];Line @(@(($x+$a[0]),($y+$a[1])),@(($x+$z[0]),($y+$z[1]))) $color 2}
    foreach($n in $nodes){Dot ($x+$n[0]) ($y+$n[1]) 4 $color}
  }
  if($trajectory){
    $p=[System.Drawing.Pen]::new((Col $color),4)
    $p1=[System.Drawing.PointF]::new(($x+10),($y+65));$p2=[System.Drawing.PointF]::new(($x+42),($y+31));$p3=[System.Drawing.PointF]::new(($x+111),($y+49));$p4=[System.Drawing.PointF]::new(($x+127),($y+11))
    $g.DrawBezier($p,$p1,$p2,$p3,$p4);$p.Dispose()
    $d="M $($x+10) $($y+65) C $($x+42) $($y+31), $($x+111) $($y+49), $($x+127) $($y+11)"
    [void]$svg.AppendLine("<path d='$d' fill='none' stroke='$color' stroke-width='4' stroke-linecap='round'/>")
    Dot ($x+10) ($y+65) 5 $color;Dot ($x+127) ($y+11) 5 $color
  }
}
function Heatmap($x,$y) {
  $palette=@('#E9F3CF','#B8DEAE','#78C6B6','#45A9B5','#3486AC','#36669B')
  for($row=0;$row -lt 4;$row++){for($col=0;$col -lt 7;$col++){
    $idx=[Math]::Min(5,[Math]::Max(0,[Math]::Floor(($col+$row)/1.7)))
    Box ($x+$col*20) ($y+$row*19.5) 20 19.5 $palette[$idx] '#FFFFFF'
  }}
  Line @(@(($x-5),($y-3)),@(($x-5),($y+83)),@(($x+145),($y+83))) '#8DA2B9' 1.5
}
function RobotFailure($x,$y) {
  Line @(@($x,($y+65)),@(($x+55),($y+65))) '#8294A8' 5
  Line @(@(($x+28),($y+64)),@(($x+28),($y+38)),@(($x+77),($y+12)),@(($x+119),($y+34))) '#557790' 8
  Dot ($x+28) ($y+38) 7 '#FFFFFF' '#557790'
  Dot ($x+77) ($y+12) 10 '#FFF2E9' '#D97C49'
  Line @(@(($x+72),($y+7)),@(($x+82),($y+17))) '#D97C49' 2.5
  Line @(@(($x+82),($y+7)),@(($x+72),($y+17))) '#D97C49' 2.5
  Line @(@(($x+119),($y+26)),@(($x+119),($y+42)),@(($x+133),($y+42))) '#557790' 3
}
Txt 60 45 1800 72 'Reliability-Aware Motion Planning Framework' 44 '#172B46' $true 'Near'
Box 385 195 1130 390 '#F2F6FC' '#CBD9EC'
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
  Box $xs[$i] 285 235 210 '#FFFFFF' '#9CB7D7'
  Txt ($xs[$i]+12) 290 211 35 ('0'+($i+1)) 19 '#4975A3' $true 'Near'
  $textSize=24; if($i -eq 2){$textSize=22}
  Txt ($xs[$i]+8) 402 219 85 $labels[$i] $textSize '#172B46' $true
}
Region 468 318
Heatmap 743 318
Region 1018 318 $true
Region 1293 318 $false $true
Arrow @(@(655,380),@(695,380)) '#4975A3'
Arrow @(@(930,380),@(970,380)) '#4975A3'
Arrow @(@(1205,380),@(1245,380)) '#4975A3'
Arrow @(@(537,495),@(537,550),@(1087,550),@(1087,495)) '#4975A3'
Txt 630 550 370 35 'Collision-free region' 20 '#4975A3'
Box 420 730 235 165 '#FFFFFF' '#8ABDAC'
RobotFailure 468 744
Txt 428 819 219 70 "Joint failure`noccurs" 25 '#18694F' $true
Box 805 730 295 165 '#FFFFFF' '#8ABDAC'
Region 882 740 $true $false '#27866D'
Txt 815 820 275 70 "Use collision-free`nregion as graph" 24 '#18694F' $true
Box 1245 730 235 165 '#FFFFFF' '#8ABDAC'
Region 1293 740 $false $true '#27866D'
Txt 1255 819 215 70 "Optimize`ntrajectory" 25 '#18694F' $true
Arrow @(@(655,795),@(805,795)) '#27866D'
Arrow @(@(1100,795),@(1245,795)) '#27866D'
Arrow @(@(537,550),@(537,614),@(952,614),@(952,730)) '#27866D' $true
Box 671 594 300 40 '#FFFFFF' '#FFFFFF'
Txt 675 594 292 40 'Reuse collision-free region' 21 '#18745E' $true
Box 1595 410 285 350 '#EAF2FB' '#90ABCA'
Txt 1610 427 255 45 'OUTPUT' 24 '#4975A3' $true
Region 1668 488 $false $true '#27866D'
Txt 1605 589 265 140 "Collision-free`ntrajectory with`nhigh reliability" 29 '#172B46' $true
Arrow @(@(1480,380),@(1545,380),@(1545,500),@(1595,500)) '#4975A3'
Arrow @(@(1480,795),@(1545,795),@(1545,610),@(1595,610)) '#27866D'
[void]$svg.AppendLine('</svg>')
[System.IO.File]::WriteAllText((Join-Path $outDir 'project_framework_illustrated.svg'),$svg.ToString(),[System.Text.UTF8Encoding]::new($false))
$bmp.Save((Join-Path $outDir 'project_framework_illustrated.png'),[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose();$bmp.Dispose()
Get-Item (Join-Path $outDir 'project_framework_illustrated.svg'),(Join-Path $outDir 'project_framework_illustrated.png') | Select-Object FullName,Length
