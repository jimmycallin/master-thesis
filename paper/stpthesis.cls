\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{stpthesis}[2011/05/16 STP thesis class]

% 9pt, 10pt, 11pt, 12pt, 14pt, 17pt
% These should essentially just be passed to memoir, but we want
% to add a different default.  I can't just do
%
%	\LoadClass[..., 11pt, ...]{memoir}
%
% because it turns out that memoir will use the *largest* size option
% given.
%
\def\stp@pointsize{11pt}
\DeclareOption{9pt}{\def\stp@pointsize{9pt}}
\DeclareOption{10pt}{\def\stp@pointsize{10pt}}
\DeclareOption{11pt}{\def\stp@pointsize{11pt}}
\DeclareOption{12pt}{\def\stp@pointsize{12pt}}
\DeclareOption{14pt}{\def\stp@pointsize{14pt}}
\DeclareOption{17pt}{\def\stp@pointsize{17pt}}


% What fonts to use?
\newif\if@stp@times\@stp@timesfalse
\newif\if@stp@uufonts\@stp@uufontstrue
\DeclareOption{cm}{\@stp@uufontsfalse\@stp@timesfalse}
\DeclareOption{times}{\@stp@uufontsfalse\@stp@timestrue}
\DeclareOption{uufonts}{\@stp@timesfalse\@stp@uufontstrue}

% Now obligatory, since the previous default, Master's, wasn't good.
\newif\if@stp@bachelormasterset\@stp@bachelormastersetfalse
% Bachelor's or Master's?
\newif\if@stp@bachelor\@stp@bachelorfalse
\DeclareOption{bachelor}{\@stp@bachelortrue\@stp@bachelormastersettrue}
\DeclareOption{master}{\@stp@bachelorfalse\@stp@bachelormastersettrue}

\newif\if@stp@color\@stp@colortrue
\DeclareOption{color}{\@stp@colortrue}
\DeclareOption{bw}{\@stp@colorfalse}

% Most users probably won't bother fixing overfull boxes anyway.
% If you want to, then use "draft".
\AtBeginDocument{%
	\ifdraftdoc
	\else
          \sloppy
        \fi}

% Continuous equation numbers in whole document
\newif\if@stp@conteqno \@stp@conteqnotrue
\DeclareOption{eqnobychapter}{\@stp@conteqnofalse}

\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{memoir}}

\ProcessOptions\relax

% Fonter
% UU:s grafiska profil säger
% (på <http://info.uu.se/profil.nsf/sidor/teckensnitt.id76.html>)
%
% Trycksaker och annat tryckt externt informationsmaterial:
%   Gill Sans MT, Berling
% Kontorstryck:
%   Arial, Times
% På webben:
%   Verdana, Arial

% UU:s rapportmall har
%	Framsida	Titel		Arial 35 pt
%			Undertitel	Arial 22 pt
%			Avsändare	Arial 12 pt
% Dessutom arial för innehållsförteckning, sidhuvud,
% sidfot, rubriker, tabeller
% Times New Roman för brödtext (10,5 pt)

% The Ariel trademark is owned by Monotype Imaging, but Microsoft
% retains exclusive license rights to Arial Unicode MS.

% pslatex använder Times som rmdefault och phv = helvetica
% som sfdefault. Det får duga.
\if@stp@times
	\usepackage{pslatex}
\else
  \if@stp@uufonts
  	\usepackage{uufonts}
  \fi
\fi
\frenchspacing

% The error is not really there, but better than to say that it's in
% this file.
\AtBeginDocument{%
\if@stp@bachelormasterset\else
  \ClassError{stpthesis}%
    {Neither [bachelor] or [master] used}%
    {Use \protect\documentclass[bachelor]{stpthesis} for a Bachelor's Thesis}
\fi
}

\if@stp@conteqno
  \AtBeginDocument{%
	\usepackage{remreset}
	\@removefromreset{equation}{chapter}
	\renewcommand \theequation {\@arabic\c@equation}}
\fi

% Check for suggested packages to use with this.
% (Earlier this loaded these by itself, but on second thought
% I think it's better if that is in every document.
% (At least inputenc.))

\newif\if@stp@swedish \@stp@swedishfalse
\AtBeginDocument{%
	\@ifpackageloaded{inputenc}{}{%
	  \ClassWarningNoLine{stpthesis}%
	    {Probably you should use
              \protect\usepackage[utf8]{inputenc}
              or \protect\usepackage[latin1]{inputenc}}}%
	\@ifpackageloaded{babel}{%
	  \iflanguage{swedish}{\@stp@swedishtrue}{%
	    \iflanguage{english}{}{%
	      \iflanguage{american}{}{%
		{\ClassWarningNoLine{stpthesis}%
		  {Main language is not Swedish or English}}}}}}%
	  {\ClassWarningNoLine{stpthesis}%
	    {Probably you should use babel}}}

% This doesn't work, because fontenc is an exception:
%
% "There exists one package that can't be tested with the above
%  commands: the \texttt{fontenc} package pretends that it was never
%  loaded to allow for repeated reloading with different options (see
%  \texttt{ltoutenc.dtx} for details)."
%
% \AtBeginDocument{%
%   \@ifpackagewith{babel}{swedish}{%
%     \@ifpackagewith{fontenc}{T1}{}{%
%       \ClassWarningNoLine{stpthesis}
%         {Eftersom du skriver på svenska borde du nog använda
% 	  \protect\usepackage[T1]{fontenc} för att få avstavning
% 	  av ord med åäö att fungera}}}}

% do this automatically anyway
\usepackage[T1]{fontenc}
\usepackage{textcomp}

\LoadClass[a4paper, \stp@pointsize, oneside, openany]{memoir}
\RequirePackage{microtype}

\newlength{\stp@templ}
\setlength{\stp@templ}{\stockwidth}
% Assume it will be stapled on the left, which means a part of the
% page won't be seen
%   Maybe that doesn't make sense for onepage output printed on
%   both sides anyway, which often is the case, but at least that
%   much is saved for the smallest margin then.
\addtolength{\stp@templ}{-8mm}
\settrimmedsize{\stockheight}{\stp@templ}{*}

% I didn't get it to work with setting the text format
% AtBeginDocument (when we surely know what font is in use).
% So instead do it now.  We have options for choosing fonts
% anyway, so most of the time we will know now.

\setlrmargins{*}{*}{2}
\setulmarginsandblock{\spinemargin}{*}{1.618034} % golden ratio
% \checkandfixthelayout
% No, just do the first two steps to avoid lots of typeout
  \checkthelayout
  \fixthelayout
  % \typeoutlayout

% Style:
\makepagestyle{stpthesis}
\if@twoside
\copypagestyle{stpthesis}{headings}
\else
% \makeoddhead{stpthesis}{\slshape\rightmark}{}{}
% \makeoddfoot{stpthesis}{}{\thepage}{}
\copypagestyle{stpthesis}{plain}
\fi
\pagestyle{stpthesis}

\makechapterstyle{stp}{%
  \renewcommand{\printchaptername}{}
  \renewcommand{\chapternamenum}{}
  \renewcommand{\chapnumfont}{\normalfont\huge\sffamily}
  \renewcommand{\chaptitlefont}{\chapnumfont}
  \renewcommand{\printchapternum}{\chapnumfont \thechapter\quad}
  \renewcommand{\afterchapternum}{}

  \renewcommand{\secheadstyle}{\LARGE\sffamily\raggedright}
  \renewcommand{\subsecheadstyle}{\Large\sffamily\raggedright}
  \renewcommand{\subsubsecheadstyle}{\large\sffamily\bfseries\raggedright}
}
\chapterstyle{stp}
\maxsecnumdepth{subsection}

% "rapport"-stilen har ungefär så här:
\captionnamefont{\small\sffamily\bfseries}
\captiontitlefont{\small\sffamily}
% Fast dom har själva tabellerna också med sanserif.

% Avhandlingar hur?

% \captionnamefont{\small\itshape}
% \captiontitlefont{\small\itshape}


% Från "Examensarbete på STP", Mats Dahllöf, 2002-11-25:
% "Examensarbetare på STP ska alltid ha en huvudhandledare från
%  Institutionen för lingvistik. Huvudhandledaren är en av
%  institutionens disputerade lärare, ..." Om examensarbetet
%  görs utanför institutionen, finns det ofta också en bihandledare
%  på den inrättning eller det företag det handlar om."

\newcommand{\credits}[1]{\gdef\@stp@credits{#1}}

\newcommand{\supervisor}[1]{\gdef\@supervisor{#1}}
\newcommand{\supervisors}[1]{\gdef\@supervisors{#1}}
% "bihandledare" **
\newcommand{\assistantsupervisor}[1]{\gdef\@assistantsupervisor{#1}}
\newcommand{\assistantsupervisors}[1]{\gdef\@assistantsupervisors{#1}}

\newcommand{\authoremail}[1]{\gdef\@authoremail{#1}}

\newcommand{\subtitle}[1]{\gdef\@subtitle{#1}}

\aliaspagestyle{title}{empty}

% Title page
%
% Strömquist recommends the SIS standard that is in SIS handbok 126,
% _Skriva på kontor_ (1995).
%
% It's hard to combine that with the UU logo though.

\RequirePackage{graphicx}

\renewcommand{\maketitle}{%
  \thispagestyle{title}%
  \bgroup%
  \parindent=0pt
  \vglue-\headheight\vglue-\headsep\vglue-\topmargin\vglue-1in%
  \hglue-\oddsidemargin\hglue-1in\hglue-\parindent%
  \sffamily
  \hbox{%
    \raisebox{0pt}[42mm][0pt]{\vtop to 293mm{%
        \if@stp@color
        \hbox{\includegraphics{images/UU_logo_pc_2f_42}}%
        \else
        \hbox{\includegraphics{images/UU_logo_pc_sv_42}}%
        \fi
        \vfil
        \hbox{\includegraphics[width=42mm]{images/UU_marginalsigill_5Svart}}}}%
    % \raisebox{42mm}[0pt][0pt]{\vrule depth \paperheight}%
    \raisebox{44mm}[0pt][0pt]{\vrule depth \paperheight}%
    \hspace{8mm}%
    \raisebox{0mm}[0pt][0pt]{%
      \vtop to 225mm{%
        \hsize=111mm            % 112 yields overfull box
        \vskip 40\p@
        \begin{flushleft}
          \fontsize{25}{30}\selectfont\textbf{\@title} \par
          \ifx\@subtitle\undefined
          \else
          \vskip 6mm
          \fontsize{17}{22}\selectfont\textit{\@subtitle}
          \fi
          \vskip 14mm
          \fontsize{17}{22}\selectfont\theauthor\par
        \end{flushleft}%
        \vfil
        \fontsize{11}{13.6}\selectfont
        \if@stp@swedish
        Institutionen f\"or lingvistik och filologi\\
        \if@stp@bachelor
        Spr{\aa}kteknologiprogrammet\\
        Kandidatuppsats
        \else
        Masterprogrammet i spr{\aa}kteknologi\\
        Masteruppsats
        \fi
        i spr{\aa}kteknologi%
        \ifx\@stp@credits\undefined
        \else
        , \@stp@credits\,hp
        \fi
        \else
        % (Include the name of the university since it's only
        % in Swedish in the logo.)
        Uppsala University\\
        Department of Linguistics and Philology\\
        \if@stp@bachelor
        Spr{\aa}kteknologiprogrammet\\
        (Language Technology Programme)\\
        Bachelor's
        \else
        Master's Programme in Language Technology\\
        Master's
        \fi
        Thesis in Language Technology%
        \ifx\@stp@credits\undefined
        \else
        , \@stp@credits\,credits
        \fi
        \fi
        \\[1ex]
        \@date
        \vskip 3mm
        \ifx\@supervisors\undefined
        \ifx\@supervisor\undefined
        \else
        \par
        \if@stp@swedish Handledare:\else Supervisor:\fi \\
        \@supervisor
        \fi
	\else
        \par
        \if@stp@swedish Handledare:\else Supervisors:\fi \\
        \@supervisors
	\fi
      }}}%
  \egroup
  \rmfamily
  \clearpage
}

% More/other Swedish texts
\AtBeginDocument{
    \@ifpackagewith{babel}{swedish}{%
     \addto\captionsswedish{%
      % Strömquist: "Om du skriver på svenska bör du använda benämningen
      %   sammandrag. Ordet sammanfattning har en tendens att bli
      %   missförstått."
       \def\abstractname{Sammandrag}
       %\def\bibname{Litteraturförteckning}
      %
      % Additional words and phrases used by memoir:
       \def\appendixtocname{Bilagor}
       \def\appendixpagename{Bilagor}
       \def\figurerefname{figur}
       \def\tablerefname{tabell}
       \def\pagerefname{sidan}
       \def\partrefname{del~}
       \def\chapterrefname{kapitel~}
      % \def\sectionrefname{\S}
      }}{}}

% Detta stämmer nu med Strömquist: "Vid pagineringen bör du, oavsett
%    om ditt arbete har pärm/omslag eller inte, börja räkna på
%    titelsidan."

% Abstract:

% This is @bstr@ctlist from memoir.dtx (version 1.6), with
% two changes:
% Add		\thispagestyle{empty}
% Set zero \itemindent instead of \listparindent.
% (= Don't indent first paragraph.)

% \begin{environment}{@bstr@ctlist}
%    The abstract's text is typeset within the |@bstr@ctlist| list environment.
%    \begin{macrocode}
\renewenvironment{@bstr@ctlist}{%
  \list{}{%
	  \thispagestyle{empty}
          %%\topsep        \z@
          \partopsep     \z@
          \listparindent \absparindent
%%%       \itemindent    \listparindent
	  \itemindent	 \z@
          \leftmargin    \abs@leftindent
          \rightmargin   \absrightindent
          \parsep        \absparsep}%
  \item\relax}
  {\endlist}

\maxtocdepth{subsection}

% Fix memoir misfeature:  Part numbers don't fit
\renewcommand{\partnumberline}[1]{%
%  \hb@xt@\@tempdima{\@cftbsnum #1\@cftasnum\hfil}\@cftasnumb}
   \@cftbsnum #1\@cftasnum\@cftasnumb\hspace{1em}}

% Interface for adjustwidth that uses all of the margins
\newenvironment{usemargins}{%
  \begin{adjustwidth}{-\spinemargin}{-\foremargin}\centering}
  {\end{adjustwidth}}

\newcounter{stp@quotelevel}
\setcounter{stp@quotelevel}{0}
\newcommand{\setquotes}[4]{%
  \def\leftquotei{#1}    \def\leftquoteii{#2}
  \def\leftquoteiii{#1}  \def\leftquoteiv{#2}
  \def\rightquotei{#4}   \def\rightquoteii{#3}
  \def\rightquoteiii{#4} \def\rightquoteiv{#3}\relax}

\setquotes{``}{`}{'}{''}
\AtBeginDocument{
  \@ifpackageloaded{babel}{%
    % Of course these are really not captions, but they behave
    % as captions in so far as they should keep the same value
    % that depends on the main language of the document.
    \addto\captionsswedish
	{\setquotes{\guillemotright}{'}{'}{\guillemotleft}}
    \addto\captionsenglish
        {\setquotes{``}{`}{'}{''}}}{}}

\newcommand{\q}[1]{%
  \ClassWarning{stpthesis}%
    {\protect\q\space is obsolete. See exempel.tex!}
  \ifcase\thestp@quotelevel
    \leftquotei \or \leftquoteii \or \leftquoteiii \or \leftquoteiv
   \else\@ctrerr\fi
  \addtocounter{stp@quotelevel}{1}%
  #1%
  \addtocounter{stp@quotelevel}{-1}%
  \ifcase\thestp@quotelevel
    \rightquotei \or \rightquoteii \or \rightquoteiii \or \rightquoteiv
   \fi}

\ifpdf
  \pdfoutput=1
  \usepackage[plainpages=false,pdfpagelabels,bookmarksnumbered]{hyperref}
  \usepackage{memhfixc}
\fi

% Hantera TAB-tecken i verbatim text med åtta stegs mellanrum.
% (Standard är annars att behandla en serie TAB-tecken som ett enda
% mellanrum.)
\tabson[8]

% Like upquote.sty, which doesn't work with memoir
\begingroup
\catcode`'=\active
\catcode``=\active
\g@addto@macro\verbatim@nolig@list
   {\let`\textasciigrave
    \let'\textquotesingle}
\endgroup

%% This is fixed now.
%% % Fix bug in memoir. See comp.text.tex thread "[memoir] adjustwidth +
%% % boxedverbatim" in February 2007.
%% \renewcommand*{\bvtoprulehook}{\hrule width\linewidth \nobreak \vskip -.1pt}
%% \renewcommand*{\bvendrulehook}{\hrule width\linewidth}
%% \bvendofpage{\hrule width\linewidth \kern-.4pt}

% covington.sty

%%% File covington.sty
%%%
%%% Michael A. Covington
%%% Artificial Intelligence Programs
%%% The University of Georgia
%%% Athens, Georgia 30602-7415 USA
%%% mcovingt@aisun1.ai.uga.edu
%%%
%%% You may copy and distribute this file freely.
%%% (If you modify it, please change its name and add your own comments.)
%%%
%%% See covington.tex for documentation.

% I have modified it to make it suitable for including in this file.
%
% \typeout{* covington.sty : Covington's linguistic macros, 2001 March 27 *}

%%%
%%% Stacked accents
%%%

% This is adapted from the \diatop macro in wsuipa.sty, which was
% originally designed by Jean-Pierre Paillet.

%** NO
\def\twoacc[#1|#2]{\leavevmode{\setbox1=\hbox{{#1{}}}%
                     \setbox2=\hbox{{#2{}}}%
                     \dimen0=\ifdim\wd1>\wd2\wd1\else\wd2\fi%
                     \dimen1=\ht2\advance\dimen1by-0.8ex%
                     \setbox1=\hbox to1\dimen0{\hss#1\hss}%
                     \rlap{\raise1\dimen1\box1}%
                     \hbox to1\dimen0{\hss#2\hss}}}%

% Example of usage: \twoacc[\^|\={a}]
% Special cases:

\def\acm#1{\twoacc[\'|\={#1}]}   % acute over macron
\def\grm#1{\twoacc[\`|\={#1}]}   % grave over macron
\def\cim#1{\twoacc[\^|\={#1}]}   % circumflex over macron

%%%
%%% Example numbering
%%%

% Example numbering uses the same counter as LaTeX's equation numbering.
% Thus, you can use \label and \ref to refer to numbered examples in
% exactly the same way as with equations.

% \exampleno  increments the counter and gives you its current value.
%             You can use it anywhere to generate an example number.
%
%             For example, to display a sentence with a number at the
%             right margin, do this:
%
%             \begin{flushleft}
%             This is a sentence.  \hfill (\exampleno)
%             \end{flushleft}

\newcommand{\exampleno}{\refstepcounter{equation}\theequation}

%%%
%%% The 'example' environment
%%%

% example     is an environment for displaying a single numbered
%             example, preceded by its number.  Example of use:
%
%             \begin{example}
%             This is an example.
%             \end{example}
%
%             This is actually a special case of 'examples' where
%             the first \item is automatically supplied.
%
%             To display sub-examples with numbers such as
%             (3)(a), (3)(b), etc., you can use 'itemize', thus:
%             \begin{example}
%              \begin{itemize}
%              \item[(a)] This is the first example.
%              \item[(b)] This is the second example.
%              \end{itemize}
%             \end{example}

\newenvironment{example}{\begin{examples}\item}{\end{examples}}

%%%
%%% The 'examples' environment
%%%

% examples    is an environment for displaying a series of examples
%             together, each with its own example number.  Usage:
%
%             \begin{examples}
%             \item This is the first example.
%             \item This is the second example.
%             \end{examples}
%
% Thanks to Darrel Hankerson of Auburn University, and numerous others
% across the net, for help with this macro.

\newcounter{equationsave}          % used in trick with equation number

\newenvironment{examples}%         % define "example" environment
{%
\begin{list}{(\theequation)}%
{%
\setcounter{equationsave}{\arabic{equation}}%
\usecounter{equation}%                          % sets it to zero, unwantedly
\setcounter{equation}{\arabic{equationsave}}%   % restores previous value
\setlength{\listparindent}{0pt}%
\def\makelabel##1{##1\hfil}%       % put labels flushleft in space available
}%
\raggedright}%                     % text is flushleft, not justified
{\end{list}}

%%%
%%% Sentences with word-by-word glosses
%%%

% See covingtn.tex for full documentation.  Some examples:
%
% Displayed sentence with gloss and translation:
%
%             \gll Dit is een Nederlands voorbeeld.
%                  This is a Dutch example.
%             \glt `This is an example in Dutch.'
%             \glend
%
% Same, using bracketing where words do not correspond one-to-one:
%
%             \gll Dit is een voorbeeldje     in het Nederlands.
%                  This is a {little example} in {}  Dutch.
%             \glt `This is a little example in Dutch.'
%             \glend
%
% Same, with example number added at right:
%
%             \gll Dit is een voorbeeldje     in het Nederlands.
%                  This is a {little example} in {}  Dutch.
%             \glt `This is a little example in Dutch.' \hfill (\exampleno)
%             \glend
%
% Same, with number but no (line reserved for) translation:
%
%             \gll Dit is een voorbeeldje     in het Nederlands.
%                  This is a {little example} in {}  Dutch.
%             \gln \hfill (\exampleno)
%             \glend
%
% If you want to align 3 lines rather than two, use \glll instead of \gll.
%
% Layout is critical between \gll (or \glll) and \glt (or \gln).
% After \glt, environment is essentially the same as 'flushleft'.
%
% Thanks to Marcel R. van der Goot for permission to reproduce code.
%
% Modified 1992 Nov 8 to force single spacing regardless of
% baselinestretch.

\@ifundefined{new@fontshape}%
   {\def\@selfnt{\ifx\@currsize\normalsize\@normalsize\else\@currsize\fi}}
   {\def\@selfnt{\selectfont}}
   % Thanks to Hans-Peter Kolb for this macro.

\def\gll                 % Introduces 2-line text-and-gloss.
   {\begin{flushleft}
\vskip\baselineskip%      % Added 1992 Nov 8 to force single spacing
\def\baselinestretch{1}%
\@selfnt%
\vskip-\baselineskip%
    \bgroup
    \catcode`\^^M=12
    \twosent
   }

\def\glll                 % Introduces 3-line text-and-gloss.
   {\begin{flushleft}
\vskip\baselineskip%      % Added 1992 Nov 8 to force single spacing
\def\baselinestretch{1}%
\@selfnt%
\vskip-\baselineskip%
    \bgroup
    \catcode`\^^M=12
    \threesent
   }

\def\glt{{\hskip -\glossglue}\unhbox\gline\smallskip\newline}  % \newline was \\
                           % Introduces translation.
                           % Required even if no translation follows.

\def\gln{{\hskip -\glossglue}\unhbox\gline\hfill}
                           % Alternative to glt, when one wants to
                           % put a number but no translation.

\def\glend{\end{flushleft}}  % Ends the gloss environment.

% The following TeX code is adapted, with permission, from:
% gloss.tex: Macros for vertically aligning words in consecutive sentences.
% Version: 1.0  release: 26 November 1990
% Copyright (c) 1991 Marcel R. van der Goot (marcel@cs.caltech.edu).
% Original gloss.tex and gloss.doc are available from
% csvax.cs.caltech.edu [131.215.131.131] in pub/tex
% and many other anonymous ftp archives.

\newbox\lineone % boxes with words from first line
\newbox\linetwo
\newbox\linethree
\newbox\wordone % a word from the first line (hbox)
\newbox\wordtwo
\newbox\wordthree
\newbox\gline % the constructed double line (hbox)
\newskip\glossglue % extra glue between glossed pairs or triples
% \glossglue = 0pt plus 2pt minus 1pt % allow stretch/shrink between words
\glossglue = 5pt plus 2pt minus 1pt % allow stretch/shrink between words
\newif\ifnotdone

\let\eachwordone=\itshape
\let\eachwordtwo=\normalfont
\let\eachwordthree=\normalfont

\def\lastword#1#2#3% #1 = \each, #2 = line box, #3 = word box
   {\setbox#2=\vbox{\unvbox#2%
                    \global\setbox#3=\lastbox
                   }%
    \ifvoid#3\global\setbox#3=\hbox{#1\strut{} }\fi
        % extra space following \strut in case #1 needs a space
   }

\def\testdone
   {\ifdim\ht\lineone=0pt
         \ifdim\ht\linetwo=0pt \notdonefalse % tricky space after pt
         \else\notdonetrue
         \fi
    \else\notdonetrue
    \fi
   }

{\catcode`\^^M=12 \endlinechar=-1 % 12 = other
\gdef\getwords(#1,#2)#3 #4^^M% #1=linebox, #2=\each, #3=1st word, #4=remainder
   {\setbox#1=\vbox{\hbox{#2\strut#3 }% adds space
                    \unvbox#1%
                   }%
    \def\more{#4}%
    \ifx\more\empty\let\more=\donewords
    \else\let\more=\getwords
    \fi
    \more(#1,#2)#4^^M%
   }

\gdef\donewords(#1,#2)^^M{}%

\gdef\twosent#1^^M#2^^M% #1 = first line, #2 = second line
   {\getwords(\lineone,\eachwordone)#1 ^^M%
    \getwords(\linetwo,\eachwordtwo)#2 ^^M%
    \loop\lastword{\eachwordone}{\lineone}{\wordone}%
         \lastword{\eachwordtwo}{\linetwo}{\wordtwo}%
         \global\setbox\gline=\hbox{\unhbox\gline
                                    \hskip\glossglue
                                    \vtop{\box\wordone   % vtop was vbox
                                          \nointerlineskip
                                          \box\wordtwo
                                         }%
                                   }%
         \testdone
         \ifnotdone
    \repeat
    \egroup % matches \bgroup in \gloss
   }

\gdef\threesent#1^^M#2^^M#3^^M% #1 = first line, #2 = second line, #3 = third
   {\getwords(\lineone,\eachwordone)#1 ^^M%
    \getwords(\linetwo,\eachwordtwo)#2 ^^M%
    \getwords(\linethree,\eachwordthree)#3 ^^M%
    \loop\lastword{\eachwordone}{\lineone}{\wordone}%
         \lastword{\eachwordtwo}{\linetwo}{\wordtwo}%
         \lastword{\eachwordthree}{\linethree}{\wordthree}%
         \global\setbox\gline=\hbox{\unhbox\gline
                                    \hskip\glossglue
                                    \vtop{\box\wordone   % vtop was vbox
                                          \nointerlineskip
                                          \box\wordtwo
                                          \nointerlineskip
                                          \box\wordthree
                                         }%
                                   }%
         \testdone
         \ifnotdone
    \repeat
    \egroup % matches \bgroup in \gloss
   }
} % restore \catcode`\^^M

%\def\gloss   % raw TeX macro for 2 aligned lines
%   {\bgroup
%    \catcode`\^^M=12
%    \twosent
%   }


%%%
%%% Phrase-structure rules
%%%

% \psr         To print S -> NP VP, use: \psr{S}{NP~VP}

\newcommand{\psr}[2]{\mbox{\textit{#1 $\rightarrow$ #2}}}


%%%
%%% Feature structures
%%%

% \fs         To print a feature structure by itself, type for example
%             \fs{case:nom \\ person:P}
%
% \lfs        To print the same feature structure with the category
%             label N at the top, type:
%             \lfs{N}{case:nom \\ person:P}

%    Modified 1990 Dec 5 so that features are left aligned.
%    Modified 2001 Mar 27 so that italics are used without the user typing \it.
\newcommand{\fs}[1]
{
    \mbox{\small
          $
          % FIXME: original had "\it" here
          \left[
          \!\!
          \begin{array}{l}
          #1
          \end{array}
          \!\!
          \right]
          $}
}

%     Modified 1990 Dec 5 so that features are left aligned.
\newcommand{\lfs}[2]
   {
     \mbox{$
           % FIXME: original had "\it" here
           \!\!
           \begin{array}[t]{c}
           #1
           \\
           \mbox{\small%
                 $
                 % FIXME: original had "\it" here
                 \left[
                 \!\!
                 \begin{array}{l}
                 #2
                 \end{array}
                 \!\!
                 \right]
                 $}
           \end{array}
           \!\!
           $}
   }


%%%
%%% Discourse representation structures
%%%

% \drs        typesets a DRS in box notation a la Kamp.
%             Arguments:
%             1. List of discourse variables joined by ~
%             2. List of conditions joined by \\
%
%             Example:   \drs{X}{donkey(X)\\green(X)}
%
%             Example:   \drs{~}{purple(X,Y)}
%
%             Example:   \drs{X}
%                            {named(X,`Pedro') \\
%                             \drs{Y}{donkey(Y)\\owns(X,Y)}~~
%                             {\large $\Rightarrow$}~
%                             \drs{~}{feeds(X,Y)}
%                            }

% \sdrs       like \drs but first argument is a sentence
%             to be displayed above the drs.
%
%             Example:   \sdrs{A donkey is green.}{X}{donkey(X)\\green(X)}

% \ifdrs      forms 2 DRSes joined with a big arrow.
%             Usually used for DRSes nested within another.
%
%             Example:   \ifdrs{X}{donkey(X)\\hungry(X)}
%                              {~}{feeds(Pedro,X)}

% \alifdrs    is like \ifdrs but shifted to the left so that an
%             embedded `if' structure will line up with ordinary
%             DRS conditions.

% \negdrs     forms a DRS preceded by the negation symbol.
%
%             Example:   \negdrs{X}{donkey(X)\\green(X)}


\newcommand{\drs}[2]
{
    {
    \itshape
    \begin{tabular}{|l|}
    \hline
    ~ \vspace{-2ex} \\
    #1
    \\
    ~ \vspace{-2ex} \\
    \hline
    ~ \vspace{-2ex} \\
    #2
    \\
    ~ \\    % can't vspace here or the line will come out wrong
    \hline
    \end{tabular}
    }
}

\newcommand{\sdrs}[3]
{\begin{tabular}{l}
\mbox{\textit{#1}} \\
~ \\
\drs{#2}{#3}
\end{tabular}}

\newcommand{\ifdrs}[4]
{
  \mbox{\drs{#1}{#2}~~{\large $\Rightarrow$}~\drs{#3}{#4}}
}

\newcommand{\alifdrs}[4]
{
  \mbox{$\!\!\!$\drs{#1}{#2}~~{\large $\Rightarrow$}~\drs{#3}{#4}}
}

\newcommand{\negdrs}[2]
{
  \mbox{{\large $\neg$}\drs{#1}{#2}}
}


%
% Exercises, reference lists, and miscellany
%

% exercise    Environment for exercises, numbered by subsection.
%             Optional argument for describing the exercise
%             ("Project" or the like).

% \bx         Abbreviation for \begin{exercise}
% \ex         Abbreviation for \end{exercise}

% reflist     Environment for hanging-indented list
%             (suitable for LSA-style reference list).
%             Does NOT require \item before entries.

% \sentence   Displays its argument, flushleft, italicized.

% \either     Displays its arguments as alternatives between
%             large curly brackets.  Good for alternatives
%             within sentences or PS-rules.

\newtheorem{exercise}{Exercise}[subsection]

% Exercise numbering resets to 0 not only at the start of
% a subsection, but also at the start of a section or chapter
% even if it is not divided into subsections.
\@addtoreset{exercise}{section}
\@addtoreset{exercise}{chapter}

\@ifundefined{reflist}{
\newenvironment{reflist}{\begin{list}{}%
{\setlength{\labelwidth}{0pt}
 \setlength{\labelsep}{0pt}
 \setlength{\leftmargin}{3em}
 \setlength{\itemindent}{-3em}
 \setlength{\listparindent}{-3em}
 \setlength{\itemsep}{6pt}}\item}{\end{list}}
}
% Modified 1992 Nov 8 to skip defining reflist if reflist is already
% defined.  Permits using uga.sty with covingtn.sty.

\newcommand{\sentence}[1]
     {\begin{flushleft}\textit{#1}\end{flushleft}}

\newcommand{\either}[2]
{
     $
     \left\{
     \begin{array}{l}
     \makebox{#1} \\
     \makebox{#2} \\
     \end{array}
     \right\}
     $
}

%%%%%%%%%%%%%%%%% End of Covington's Macros %%%%%%%%%%%%%%%%%
