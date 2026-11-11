from pathlib import Path
import re,json,math
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'Report/DesignPolished_2026-11-01'
# Values below are constructed worked examples, not newly measured system results.
examples={
'eq:des-pulley':('','reproduce.py, GantryConfig; hardware records'),
'eq:des-stepres':('','reproduce.py, GantryConfig; hardware records'),
'eq:des-rpm':('','reproduce.py, GantryConfig'),
'eq:des-torque':(r'For a chosen acceleration case, $r_p=0.006366$ m, $m=0.950$ kg, $a=0.5$ m/s$^2$, $F_f=0.0373$ N and $J_{\rm rot}=10^{-5}$ kg\,m$^2$ give $T_{\rm req}=0.006366(0.0373+0.475)+10^{-5}(0.5/0.006366)=0.00405$ N\,m. The resistance and inertia are assumed; this is not the installed motor\textquotesingle s verified torque margin.','hardware records; stated load-model assumptions'),
'eq:des-pulsetime':(r'For an illustrative 10 mm move at $F=800$ mm/min with $N_M=800$, $N_m=0$, $\tau=10/[(800/60)800]=0.0009375$ s and $t_{\rm delay}=0.750$ s. A 10 mm diagonal with $N_M=N_m=566$ instead gives approximately 1.50 s of pulse delay.','reproduce.py, BresenhamMove'),
'eq:des-basecurrent':('','hardware records; ROHM low-side relation'),
'eq:des-calibration':(r'Using the configured $L_x=200.6$ mm and an illustrative counted $N_x=16048$ gives $\kappa_x=80$ microsteps/mm. A target $x=20$ mm with $e=5$ mm becomes $n_x=\operatorname{round}(25\times80)=2000$. The step count is a chosen example, not a recovered calibration measurement.','reproduce.py, Calibrate'),
'eq:des-scaleerror':('','reproduce.py, calibration geometry'),
'eq:des-current':('','A4988 datasheet, p. 9'),
'eq:des-ratio':(r'With illustrative $G=90$, $B_r=180$ and $t^*=0.70$, $R=90/180=0.50$ and $M=1$. Paper at $G=171$ under the same background gives $R=0.95$ and $M=0$.','segment.py, Deskew'),
'eq:des-integral':(r'For four prefix entries $S_{11}=900$, $S_{01}=300$, $S_{10}=200$ and $S_{00}=80$, the window sum is $900-300-200+80=480$. A four-pixel window therefore has $B_r=120$.','segment.py, Integral and BoxSum'),
'eq:des-otsu':(r'For a constructed candidate split with $\omega_0=0.2$, $\omega_1=0.8$, $\mu_0=0.3$ and $\mu_1=1.0$, the criterion is $0.2(0.8)(0.3-1.0)^2=0.0784$. A candidate giving 0.060 is inferior; the implementation searches all valid histogram splits.','segment.py, Otsu'),
'eq:des-skew':(r'Two candidate row profiles $(0,6,0,6)$ and $(3,3,3,3)$ contain the same twelve ink pixels, but score $72$ and $36$. The first is preferred because the ink is more concentrated in rows.','segment.py, Deskew'),
'eq:des-seam':(r'For local cost $C=0.10$ and predecessor pairs $(D,d)=(0.30,0),(0.20,1),(0.25,-1)$, the accumulated cost is $0.10+\min(0.30,0.22,0.27)=0.32$. The selected predecessor changes row by one.','segment.py, LeastCostSeam'),
'eq:des-seamcost':(r'With $h_g=20$ pixels, blank ink and $y-y_m=10$ pixels, $C=0.012(10/20)^2=0.003$. Crossing ink at the midpoint costs $4$, over a thousand times as much locally. These are the implemented weights, not measured probabilities.','segment.py, SeamCuts'),
'eq:des-fit':(r'A chosen $80\times1000$ crop gives $a=\min(0.8,0.64)=0.64$, hence an approximately $51\times640$ image plus white vertical padding. Its aspect ratio is retained, but its text body is shorter than 64 pixels.','segment.py, FitFixedCanvas'),
'eq:des-conv':(r'In one illustrative single-channel $3\times3$ patch, set the nine inputs to $1,\ldots,9$, all kernel weights to one and the bias to 0.5. The output is $z=0.5+45=45.5$ before normalisation and ReLU. These are illustrative weights, not learned parameters.','recognise.py, Conv2d'),
'eq:des-bn':(r'For illustrative $z=3$, $\mu_B=1$, $\sigma_B^2=4$, $\epsilon=10^{-5}$, $\gamma=1.5$ and $\beta=0.1$, $\widehat z=2/\sqrt{4.00001}\simeq1$ and $y\simeq1.6000$.','recognise.py, BatchNorm2d; train.py, BatchNorm2dForwardTrain'),
'eq:des-lstm':(r'One scalar cell with $c_{t-1}=0.5$, $f_t=0.8$, $i_t=0.3$, $g_t=0.4$ and $o_t=0.7$ gives $c_t=0.8(0.5)+0.3(0.4)=0.52$ and $h_t=0.7\tanh(0.52)=0.3344$.','recognise.py, LstmForward'),
'eq:des-ctc':(r'For a two-frame, two-symbol example targeting ``a'', let $p_1(a)=0.8$ and $p_2(a)=0.7$, with the remaining mass assigned to blank. The admissible paths $(a,a),(a,\varnothing),(\varnothing,a)$ sum to $0.56+0.24+0.14=0.94$, giving $\mathcal L=-\log(0.94)=0.06188$.','train.py, CtcLoss'),
'eq:des-ctcgrad':(r'In that example, the posterior occupancy of ``a'' at frame 1 is $q_{1,a}=(0.56+0.24)/0.94=0.85106$. Its unaveraged logit derivative is $0.8-0.85106=-0.05106$, encouraging more probability on the target symbol.','train.py, CtcLoss'),
'eq:des-strokenorm':(r'At the selected $\rho=0.17$ and an illustrative estimated $h_x=24$ pixels, $w=\operatorname{clip}(\operatorname{round}(4.08),2,9)=4$ and the actual odd kernel is $k=5$. The redraw is 0 at dilated ink pixels and 255 elsewhere.','recognise.py, StrokeNormalize'),
'eq:des-convback':(r'For a reduced two-input patch $X_c=(1,2)^T$, kernel $W=(0.1,0.2)$ and incoming derivative $G=3$, the derivatives are $GX_c^T=(3,6)$ and $W^TG=(0.3,0.6)^T$. Overlapping patches add their contributions to shared input pixels.','train.py, Conv2dBackward'),
'eq:des-gradcheck':(r'For the controlled scalar loss $\mathcal L(\theta)=\theta^2$, $\theta=2$ and $\varepsilon=10^{-4}$, the estimate is $[(2.0001)^2-(1.9999)^2]/0.0002=4$, matching $2\theta$. This demonstrates the reference calculation, not a new whole-network gradient test.','Evidence/bench.json, ctc'),
'eq:des-viterbi':(r'If $p_t(\ell_s\prime)=0.7$ and the admissible predecessor log scores are $-2$ and $-3$, $V=\log(0.7)+\max(-2,-3)=-2.3567$. Extraction selects one predecessor; the CTC likelihood instead sums both.','recognise.py, CtcForcedAlign'),
'eq:des-cutcost':(r'For illustrative smoothed ink count 2, reference percentile 10 and $(b-y_{\rm top})/h_x=0.2$, $C_x=2/10+1.15(0.2)=0.43$. At the same ink count but body height 1.0, the cost rises to 1.35.','style.py, ExtractLineGlyphs'),
'eq:des-glyphcoords':(r'With chosen $b=40$, $y=28$, $h_x=24$, $x-x_L=12$, $\tan\phi=0.2$ and post-trim $\delta=0.1$, $v=0.5$, $u_0=0.5-0.5(0.2)=0.4$ and stored $u=0.3$. The separate lead is $l=0.1$.','style.py, ExtractLineGlyphs'),
'eq:des-connectedness':(r'If twelve of twenty eligible adjacent-letter pairs cross their source boundaries, $c=12/20=0.60$. This illustrative fraction describes observed connections, not the proportion of future joins guaranteed to succeed.','style.py, BuildAuthorProfile'),
'eq:des-placement':(r'Using the preceding glyph point, choose $h_x^{\rm mm}=4$, $p_x=10$, $p_y=20$, $s=0.2$ and $d=0$. Then $x=10+4(0.1+0.3+0.2\times0.5)=12$ mm and $y=20+4(0.5)=22$ mm.','synthesise.py, SynthesizeText'),
'eq:des-anchor':(r'An illustrative median prior distance $\widetilde d=0.28$ gives $\lambda=(0.28-0.24)/0.08=0.5$; distances below 0.24 or above 0.32 saturate at zero or one.','synthesise.py, AutoLegibility'),
'eq:des-jointscore':(r'For a 20-character target with two edit operations, $A=1-2/20=0.90$. If the selected writer posterior is $P=0.80$, $J=2(0.90)(0.80)/1.70=0.8471$. This constructed score explains selection and is not a measured stylistic-correlation result.','recognise.py, CharAcc; synthesise.py, SynthesizeJointBestOf'),
'eq:des-machinecoords':(r'For a chosen origin $(0,250)$ mm, margins $(15,10)$ mm, scale $a=1$, bounds $x_0=0$, $y_1=20$, and no line shifts, trajectory point $(5,12)$ becomes $(x_m,y_m)=(20,232)$ mm. These illustrative coordinates must still pass workspace checks.','reproduce.py, ToMachine'),
'eq:des-timebudget':(r'For chosen $D_d=100$ mm, $D_u=50$ mm, $F_d=800$ mm/min, $F_u=4000$ mm/min, twenty transitions and two 0.35 s dwells, $t_{\rm nom}=7.50+0.75+7.00=15.25$ s. It remains a nominal budget, not a measured job time.','reproduce.py, WriteGcode'),
'eq:des-pagevote':(r'For two illustrative line weights 100 and 300 with probabilities 0.90 and 0.60 for writer $k$, $\overline p_k=[100(0.90)+300(0.60)]/400=0.675$. The second line has greater influence because it contains more counted ink.','web/server.py, NameWriter'),
'eq:app-bnback':(r'For three scalar inputs $(-1,0,1)$, $\gamma=1$, $\epsilon=10^{-5}$ and incoming gradients $(1,0,0)$, the middle normalised input is zero. Its derivative is $-1/[3\sqrt{2/3+10^{-5}}]=-0.40825$, despite its own incoming gradient being zero. The shared statistics couple the samples.','train.py, BatchNorm2dBackward'),
'eq:app-lstmback':(r'Using the main-text scalar cell $c_t=0.52$, $o_t=0.7$, $f_t=0.8$, with $dh_t=1$ and $dc_t^{\rm next}=0$, gives $dc_t=0.7[1-\tanh^2(0.52)]=0.5404$ and $dc_{t-1}=0.4323$, before any other accumulated paths.','train.py, LstmBackward'),
'eq:app-ctcforward':(r'For emission 0.7 and predecessor log scores $-2,-3$, $\alpha=\log(0.7)+\log(e^{-2}+e^{-3})=-2.0434$. This exceeds the max-path score $-2.3567$ because both admissible paths contribute.','train.py, CtcLoss'),
'eq:app-rmsprop':(r'At the first update choose $\theta=1$, $g=0.2$, $\lambda_w=10^{-5}$, $v_0=0$ and $\eta=10^{-3}$. Then $g\prime=0.20001$, $v_1=0.000400040001$ and $\theta_1\simeq0.990000$. The values illustrate the implemented update, not a recovered training step.','train.py, RMSprop'),
'eq:app-adamw':(r'With zero initial moments, $g=0.2$, $\theta_0=1$, $\eta=10^{-3}$ and $\lambda_w=10^{-4}$, $m_1=0.02$, $v_1=0.00004$, $\widehat m_1=0.2$, $\widehat v_1=0.04$ and $\theta_1\simeq0.9989999$.','train.py, AdamW'),
'eq:app-thinning':(r'For neighbours $(P_2,\ldots,P_9)=(1,1,1,0,0,0,0,0)$, the neighbour count is three, the circular transition count is one and all listed products are zero. A foreground centre therefore qualifies for deletion in the first subpass; the next subpass uses the updated image.','segment.py, ZhangSuenThin'),
}

def codecite(locator):
 # Plain text locators wrap naturally; no fabricated PDF pages for unpaginated code.
 if locator.startswith('hardware') or 'hardware records' in locator:
  return r'\cite[hardware clarifications]{brauneevidence2026}'
 if locator.startswith('Evidence/'):
  return r'\cite[\texttt{Evidence/bench.json}, CTC check]{brauneevidence2026}'
 if locator.startswith('A4988'):return r'\cite[p.~9]{a4988ds}'
 return r'\cite['+locator+r']{braunepipeline2026}'

files=[*sorted((B/'3_Main_Report/design').glob('*.tex')),B/'Appendices/design_implementation_details.tex']
inventory=[]
for p in files:
 t=p.read_text(encoding='utf-8')
 def annotate(m):
  key=re.search(r'\\label\{([^}]+)\}',m[0]).group(1)
  example,loc=examples[key];inventory.append({'equation':key,'file':p.relative_to(B).as_posix(),'locator':loc,'numerical_example':'adjacent added' if example else 'existing substitution or following calculation'})
  # The source of the implemented convention is directly at each equation.
  addition='\n'+('Calculated example: '+example+' ' if example else 'Implementation/input basis: ')+codecite(loc)+'\n'
  return m[0]+addition
 assert 'Calculated example:' not in t,p
 t=re.sub(r'\\begin\{equation\}.*?\\end\{equation\}',annotate,t,flags=re.S)
 # Retain unsupported page locators honestly inside the cite command itself.
 t=re.sub(r'\\cite\{(acarnley2002|bc337ds|crow1984|otsu1979|postl1986|chellapilla2006|douglas1973|marti2002)\}',r'\\cite[page to verify]{\1}',t)
 p.write_text(t,encoding='utf-8')
assert len(inventory)==40
(B/'EQUATION_AUDIT.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
bib=B/'references.bib';t=bib.read_text(encoding='utf-8')
t+=r'''
@misc{brauneevidence2026,
 author={Braune, D. R.},
 title={Handwriting robot: design records and subsystem verification evidence},
 howpublished={Unpublished project records accompanying the report source},
 year={2026},
 note={Hardware and training clarifications supplied 31 October--1 November 2026; PROJECT\_DESIGN\_RECORDS.md and Evidence manifests. Constructed examples are identified separately}
}
'''
bib.write_text(t,encoding='utf-8')
# Full precision calculation checks, retained separately from examples in the prose.
checks={'lstm_h':.7*math.tanh(.52),'lstm_dc':.7*(1-math.tanh(.52)**2),'bn_middle_grad':-1/(3*math.sqrt(2/3+1e-5)),'ctc_loss':-math.log(.94),'ctc_occupancy':.8/.94,'viterbi':math.log(.7)-2,'ctc_alpha':math.log(.7)+math.log(math.exp(-2)+math.exp(-3)),'torque':.006366*(.0373+.950*.5)+1e-5*.5/.006366}
(B/'WORKED_CALCULATION_CHECKS.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
