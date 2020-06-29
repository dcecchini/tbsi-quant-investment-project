%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Class ``tbsi-thesis''
%%
%% Version: v0.1
%% Authors: David Cecchini
%% Date: 14/04/2020
%%
%% Copyright (c) 2020-2024, David Cecchini
%% License: Simplified BSD License
%% License file: ./License
%% Original License URL: http://www.freebsd.org/copyright/freebsd-license.html
%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% General guidelines on which this class is based:
%%
%%     https://www.tsinghua.edu.cn/not/yet/known
%%
%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Class identification.
%%
%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tbsi-thesis}[2020/04/14 Tsinghua-Berkeley Shenzhen Institute thesis class]

\typeout{}
\typeout{***********************************************}
\typeout{***********************************************}
\typeout{}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% General guidelines.
%%
%% LIMIT: 60k words (including tables and footnotes, excluding appendices, bib,
%% photos, diagrams); title and section headings should be capitalized as normal
%% sentences; citations should include authors' initials, and page numbers (if
%% possible); double-sided printing is permissible for the soft bound version;
%% however, single-sided is required for the text of the final, hard bound
%% library copy (diagrams on facing pages are acceptable); always make it
%% possible to create the ps file as well (required for technical reports).
%%
%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Package options (see README.md for a list of options with descriptions).
%%
%% These options can be provided within square brackets of the `documentclass'
%% command.
%%
%%%%%

% techreport - formats the thesis as a technical report.
\newif\iftbsi@techreport\tbsi@techreportfalse
\DeclareOption{techreport}{\tbsi@techreporttrue}

% TODO
% presentation - formats the thesis as a beamer presentation.
\newif\iftbsi@presentation\tbsi@presentationfalse
\DeclareOption{presentation}{\tbsi@presentationtrue}

% TODO
% homework - formats as homeork: includes professor and course name
% in the frontmatter, ommits abstract and declaration, ommits blank pages.
\newif\iftbsi@homework\tbsi@homeworkfalse
\DeclareOption{homework}{\tbsi@homeworktrue}

% times - tells the class to use the times font.
\newif\iftbsi@times\tbsi@timesfalse
\DeclareOption{times}{\tbsi@timestrue}

% add Chinese abstract and title
\newif\iftbsi@chinese\tbsi@chinesefalse
\DeclareOption{chinese}{\tbsi@chinesetrue}

% glossary - puts the glossary (after the TOC).
%
\newif\iftbsi@glossary\tbsi@glossaryfalse
\DeclareOption{glossary}{\tbsi@glossarytrue}

% index - puts the index at the end of the thesis.
%
\newif\iftbsi@index\tbsi@indexfalse
\DeclareOption{withindex}{\tbsi@indextrue}

% 1st year report - omits abstract/declaration
%
\newif\iftbsi@firstyr\tbsi@firstyrfalse
\DeclareOption{firstyr}{\tbsi@firstyrtrue}

% 2nd year report - omits declaration
%
\newif\iftbsi@secondyr\tbsi@secondyrfalse
\DeclareOption{secondyr}{\tbsi@secondyrtrue}

% backrefs - add back references
%
\newif\iftbsi@backrefs\tbsi@backrefsfalse
\DeclareOption{backrefs}{\tbsi@backrefstrue}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Using report class as base for documents, and beamer for presentation.
%%
%%%%%
\iftbsi@presentation
  % TODO
  \PassOptionsToClass{10pt}{beamer}
  \DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
  \ProcessOptions\relax
  \LoadClass{beamer}
\else
\iftbsi@homework
  % For homework, don't need to start new sections on new page, so
  % we remove the option 'openright'
  \PassOptionsToClass{a4paper,12pt,twoside}{report}
  \DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
  \ProcessOptions\relax
  \LoadClass{report}
\else
  \PassOptionsToClass{a4paper,12pt,twoside,openright}{report}
  \DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
  \ProcessOptions\relax
  \LoadClass{report}
\fi
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% For consistent vertical spacing
%%
%%%%%
\raggedbottom


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Additional packages, and their options.
%%
%%%%%
\RequirePackage{graphicx}           % Required for the logo (on the title page)
\RequirePackage{calc}               % Used for calculating margins and laying out the title page
\RequirePackage{setspace}      	    % Used for selecting single or onehalf line spacing.
\RequirePackage{microtype}          % Improvement of general text appearance
\RequirePackage{CJKutf8}		        % for Chinese text
\RequirePackage[numbers]{natbib}    % Citations using numbers
\AtBeginDvi{\input{zhwinfonts}}	   	% Chinese fonts
\RequirePackage[scaled]{beramono}   % Fonts for source code: truetype: Bistream Vera Sans Mono
\RequirePackage{amsmath}            % For align environment and equations

% Hyperref should be included in any case, but we make it non-obtrusive (i.e.: no hyperlink borders).
% Why do we need a separate treatment of hyperref for the techreport? Because the guidelines say so. :)
\iftbsi@techreport
    % TODO: If the thesis is compiled to PS, include the 'dvips' option to hyperref.
    \RequirePackage[
        pdfpagelabels=false,
        pdfborder={0 0 0},
        pdfusetitle=true
    ]{hyperref}
\else
    \RequirePackage[
        pdffitwindow=true,
        pdfpagelabels=true,
        colorlinks=false,
        pdfborder={0 0 0},
        pdfusetitle=true
    ]{hyperref}
\fi
\iftbsi@presentation
  \RequirePackage{float}            % Adds option [H] to images
\fi
\RequirePackage[all]{hypcap}        % Fixes links to floats (must be included after hyperref).

% Create the index
\iftbsi@index
    \RequirePackage{makeidx}
    \makeindex
    \newcommand{\printthesisindex}{%
        \cleardoublepage%
        \phantomsection%
        \addcontentsline{toc}{chapter}{Index}%
        \printindex}
\fi

% Create the glossary
\iftbsi@glossary
    \RequirePackage{glossaries}
    \makeglossaries%
    \newcommand{\printthesisglossary}{\printglossary[nonumberlist]}
    \newcommand{\tbsi@printthesisglossary}{%
        \cleardoublepage%
        \pagestyle{empty}%
        \renewcommand{\glossarypreamble}{\thispagestyle{empty}}%
        \printthesisglossary%
    }
\fi


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Page margins (suitable for J.S. Wilson & Son).
%%
%%%%%

\iftbsi@presentation
\else
  \newlength{\tbsi@topmargin}
  \newlength{\tbsi@bottommargin}
  \newlength{\tbsi@oddmargin}
  \newlength{\tbsi@evenmargin}
\fi

%% Calculate and set the margins properly (with parameters that actually have
%% some meaning for everyday thesis-writers).
%%
%% @param	1	odd side margin (inner margin).
%% @param	2	even side margin (outer margin).
%% @param	3	top margin.
%% @param	4	bottom margin.
\DeclareRobustCommand{\tbsi@calcpaperdims}[4]{%
% MARGINS
% 'Top margin' is the distance between the top of the text and the top of the page.
% 'Bottom margin' is the distance between the bottom of the footer (the page number) and the bottom of the page.
\setlength{\tbsi@oddmargin}{#1}        % inner margin
\setlength{\tbsi@evenmargin}{#2}       % outer margin
\setlength{\tbsi@topmargin}{#3}        % top margin        (the distance from the top of the page to the top of the body text -- the header is located between)
\setlength{\tbsi@bottommargin}{#4}     % bottom margin     (the distance from the bottom of the page to the bottom of the body text -- the footer is located between)
% Horizontal spacing
\setlength{\textwidth}{\paperwidth-\tbsi@oddmargin-\tbsi@evenmargin}     % text takes the remaining width (210 - inner - outer)
\setlength{\oddsidemargin}{\tbsi@oddmargin-1in}   % Counter the LaTeX 1in margin
\setlength{\evensidemargin}{\tbsi@evenmargin-1in} % Counter the LaTeX 1in margin
\setlength{\marginparwidth}{\tbsi@evenmargin-8mm} % the margin only has 'outer' space available, so we have to make it a bit thinner.
\setlength{\marginparsep}{3mm}

% Vertical spacing
\setlength{\headheight}{5mm}      % The height of the box where the heading text lives
\setlength{\headsep}{5mm}         % The distance between the heading and the top of the text
\setlength{\topmargin}{\tbsi@topmargin-\headheight-\headsep-1in} % Counter the LaTeX 1in margin

\setlength{\textheight}{\paperheight-\tbsi@topmargin-1.7\tbsi@bottommargin}   % text takes the remaining height (297 - top margin - bottom margin)
\setlength{\footskip}{.7\tbsi@bottommargin} % The distance from the bottom of the text to the bottom of the footer
}
\iftbsi@presentation
  % Don't change the beamer defaults
\else
\iftbsi@techreport
\tbsi@calcpaperdims{25mm}{25mm}{20mm}{20mm}
\else
\tbsi@calcpaperdims{30mm}{20mm}{20mm}{20mm}
\fi
\fi


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Fonts
%%
%%%%%

% Font recommended by the Computer Laboratory (Adobe Sabon).
% Note: You might need to install it first.
% \renewcommand\rmdefault{psb}

% Use the times font if the user selected this option.
\iftbsi@times
    \RequirePackage{times}
\fi


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Line spacing
%%
%%%%%

% One-and-a-half spacing (required by board of graduate studies).
% (Should remove this for Computer Laboratory technical report.)
\iftbsi@techreport\else%
    \onehalfspacing%
\fi


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Variable definitions and default values: these variables should be defined by
%% the user (somewhere in the preamble). For example, to put the abstract into
%% the thesis, the thesis writer should type the following somewhere in the
%% preamble (before the `\begin{document}` or `\frontmatter` commands are
%% called):
%%
%%     \abstractCN{我的摘要。。。}
%%		 \abstractEN{This is my abstract.}
%%
%% See below (in the comments starting with 'DOCVAR: ') for a list of all
% variables
%% the thesis writer is expected to use.
%%
%%%%%

% DOCVAR: abstract (The text that will be instered into the abstract of the
% thesis.)
\newcommand{\@abstractEN}{}
\newcommand{\abstractEN}[1]{\renewcommand{\@abstractEN}{#1}}

\newcommand{\@abstractCN}{}
\newcommand{\abstractCN}[1]{\renewcommand{\@abstractCN}{#1}}

% DOCVAR: acknowledgements (The text that will be instered into the
% acknowledgments of the thesis.)
\newcommand{\@acknowledgements}{}
\newcommand{\acknowledgements}[1]{\renewcommand{\@acknowledgements}{#1}}

% DOCVAR: college (The name of the thesis writer's college, which will appear
% just below their name.)
\newcommand{\@college}{}
\newcommand{\college}[1]{\renewcommand{\@college}{#1}}

% DOCVAR: keywords (These keywords will appear in the PDF meta-information
% called `pdfkeywords`.)
\newcommand{\@keywordsEN}{}
\newcommand{\keywordsEN}[1]{\renewcommand{\@keywordsEN}{#1}}

\newcommand{\@keywordsCN}{}
\newcommand{\keywordsCN}[1]{\renewcommand{\@keywordsCN}{#1}}

% Title in Chinese for abstract page in Chinese
\newcommand{\@titleCN}{}
\newcommand{\titleCN}[1]{\renewcommand{\@titleCN}{#1}}

% DOCVAR: professor (for presentation or homework)
\newif\iftbsi@professor\tbsi@professorfalse
\newcommand{\@professor}{}
\newcommand{\professor}[1]{\renewcommand{\@professor}{#1}\tbsi@professortrue}

% DOCVAR: class name (for presentation or homework)
\newif\iftbsi@classname\tbsi@classnamefalse
\newcommand{\@classname}{}
\newcommand{\classname}[1]{\renewcommand{\@classname}{#1}\tbsi@classnametrue}

% DOCVAR: subjectline (This subject will appear in the PDF meta-information
% called `pdfsubject`.)
\newcommand{\@subjectline}{}
\newcommand{\subjectline}[1]{\renewcommand{\@subjectline}{#1}}

% DOCVAR: submissiondate (The date of the submission of this thesis. If the
% submission date is provided, it will be printed on the title page--within the
% `submissionnotice` by default. Note that the thesis writer can provide their
% own `submissionnotice`, in which case it is up to them whether they will use
% this date in their notice.)
\newif\iftbsi@submissiondate\tbsi@submissiondatefalse
\newcommand{\@submissiondate}{}
\newcommand{\submissiondate}[1]{\renewcommand{\@submissiondate}{#1}\tbsi@submissiondatetrue}

% DOCVAR: submissionnotice (The submission notice is shown on the bottom of the
% title page.)
\newcommand{\@submissionnotice}{%
\iftbsi@techreport%
    % Just the date
    \@submissiondate{}
\else
    \iftbsi@firstyr
    First year report submitted
    \else
    \iftbsi@secondyr
    Second year report submitted
    \else
    This dissertation is submitted
    \fi
    \fi
    \iftbsi@submissiondate
        on \@submissiondate{}
    \fi
    \iftbsi@firstyr
    in partial fulfilment of the requirements
    \fi
    \iftbsi@secondyr
    in partial fulfilment of the requirements
    \fi
    for the degree of Doctor of Philosophy%
\fi
}
\newcommand{\submissionnotice}[1]{\renewcommand{\@submissionnotice}{#1}}

% DOCVAR: collegeshield (The name of the file that contains the image of the
% college's shield. If `collegeshield' is provided, it will be included in the
% title page (just below the author's name and above the name of the college).
\newif\iftbsi@collegeshield\tbsi@collegeshieldfalse
\newcommand{\@collegeshield}{}
\newcommand{\collegeshield}[1]{\renewcommand{\@collegeshield}{#1}\tbsi@collegeshieldtrue}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Chapter and section numbering
%%
% TODO: Check whether numbering up to the fourth level is okay. It is not
% specified in the guidelines.
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Front matter
%%
%% - outside and inside front cover
%% - title leaf
%% Do not include the date of make!
%% Institution + department.
%% Names of referees. (optional)
%% Degree.
%% Date of submission and defense. (optional)
%% Place and date of publication and publishers (and other info by them).
%%%%%
\newcommand{\frontmatter}{

\pagestyle{empty}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Title page components
%%
%%%%%
%\iftbsi@techreport
% Technical report mustn't have the custom title page (a standard one will be
% prepended by the editor, see http://www.cl.cam.ac.uk/techreports/submission.html).
%\else
% The boxes below are all that will be displayed on the title page. They are
% used to calculate exactly how much space should be left between them
% (vertically).

%% LOGO box
\newlength{\tbsi@logorightnudge}
\setlength{\tbsi@logorightnudge}{-0.5\paperwidth+12mm}
\newsavebox{\tbsi@logo}
\begin{lrbox}{\tbsi@logo}
\hspace*{\tbsi@logorightnudge}\includegraphics[width=73mm]{images/Tsinghua}
\end{lrbox}

%% THESIS TITLE box
\newsavebox{\tbsi@title}
\begin{lrbox}{\tbsi@title}
    \begin{minipage}[c][\height][c]{.98\textwidth}
        \begin{center}
			\Huge%
				\iftbsi@times\else%
				\bfseries%
				\fi%
			{\@title{}}%
			\iftbsi@firstyr\\%
				{\vspace{5mm}\emph{\LARGE PhD Proposal}}%
			\fi%
			\iftbsi@secondyr\\%
				{\vspace{5mm}\emph{\LARGE Dissertation Schedule}}%
			\fi
		\end{center}
    \end{minipage}
\end{lrbox}

%% COLLEGESHIELD box (optional):
\iftbsi@collegeshield%
    \newsavebox{\tbsi@collegeshieldbox}
    \begin{lrbox}{\tbsi@collegeshieldbox}
    \includegraphics[height=20mm]{\@collegeshield}
    \end{lrbox}
\fi

%% AUTHOR&COLLEGE box
\newsavebox{\tbsi@authorcollege}
\begin{lrbox}{\tbsi@authorcollege}
    \begin{minipage}[c][\height][c]{.98\textwidth}
        \begin{center}
        {\large \@author{}~\\[1ex]}
        \iftbsi@collegeshield%
            \vspace{2mm}{\usebox{\tbsi@collegeshieldbox}}\\
        \fi
        \@college{}
        \end{center}
    \end{minipage}
\end{lrbox}

%% SUBMISSION NOTICE box
\newsavebox{\tbsi@submitnotice}
\begin{lrbox}{\tbsi@submitnotice}
    \begin{minipage}[c][\height][c]{.98\textwidth}
        \begin{center}
            \@submissionnotice{}
        \end{center}
    \end{minipage}
\end{lrbox}

% Now calculate the exact free vertical space
\newlength{\tbsi@titlepagevspace}
\setlength{\tbsi@titlepagevspace}{\textheight%
	-\totalheightof{\usebox{\tbsi@logo}}%
	-\totalheightof{\usebox{\tbsi@submitnotice}}%
	-\totalheightof{\usebox{\tbsi@authorcollege}}%
	-\totalheightof{\usebox{\tbsi@title}}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Title Page: Put the components (logo, title, author, college and submit
%% notice) onto the title page.
%%
%%%%%
\begin{center}
~\vspace{.02\tbsi@titlepagevspace}\\
{\usebox{\tbsi@logo}}\\
\vspace{.28\tbsi@titlepagevspace}
{\usebox{\tbsi@title}}\\
\vspace{.23\tbsi@titlepagevspace}
{\usebox{\tbsi@authorcollege}}\\
\null\vfill
{\usebox{\tbsi@submitnotice}}
\end{center}
\hypersetup{pdfsubject={\@subjectline},pdfkeywords={\@keywordsEN}}
%\fi

% Epigraph on odd page. (optional)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Declaration
%%
%%%%%
\iftbsi@techreport
% Technical report doesn't need the declaration
% (see http://www.cl.cam.ac.uk/techreports/submission.html).
\else
{\iftbsi@firstyr
% First and second yr report don't need the declaration
\else
\iftbsi@secondyr
%
\else
\chapter*{Declaration}
\thispagestyle{empty}
This dissertation is the result of my own work and includes nothing which is the outcome
of work done in collaboration except as declared in the Preface and specified in the text.
It is not substantially the same as any that I have submitted, or am concurrently
submitting, for a degree or diploma or other qualification at the Tsinghua-Berkeley Shenzhen Institute
or any other University or similar institution except as declared in the Preface and
specified in the text. I further state that no substantial part of my dissertation has
already been submitted, or is being concurrently submitted, for any such degree, diploma
or other qualification at the Tsinghua-Berkeley Shenzhen Institute or any other University or similar
institution except as declared in the Preface and specified in the text.
This dissertation does not exceed the prescribed limit of 60\,000 words.

% Leaving some space for the signature:
\vspace{15mm}

\begin{flushright}
\@author{}\\
\@date{}\\
\end{flushright}
\vfill
\fi
\fi}
\fi


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Abstracts
%%
%%%%%
\iftbsi@techreport%
    \setcounter{page}{3}
\fi

\iftbsi@firstyr
% First yr report doesn't need a standalone abstract
\else

%%%%%%%%%%%%
%%
%% CN
\iftbsi@chinese%
	\chapter*{摘要}
	\thispagestyle{empty}
	% Cambridge thesis submission guidelines require the title and author be in the abstract.
	% For more info see https://www.cambridgestudents.cam.ac.uk/your-course/examinations/graduate-exam-information/after-examination/degree-approval-and-1
	\textbf{\large \@titleCN}
	\par\vspace{0.3cm}
	\noindent\textit{\@author}
	\par\vspace{0.6cm}

	\@abstractCN{}

	% Add keywords
	\par\vspace{0.3cm}
	\textbf{关键词}: \@keywordsCN{}
\fi
%%%%%%%%%%%%
%%
%% EN
\chapter*{Abstract}
\thispagestyle{empty}

% Cambridge thesis submission guidelines require the title and author be in the abstract.
% For more info see https://www.cambridgestudents.cam.ac.uk/your-course/examinations/graduate-exam-information/after-examination/degree-approval-and-1
\textbf{\large \@title}
\par\vspace{0.3cm}
\noindent\textit{\@author}
\par\vspace{0.6cm}

\@abstractEN{}

% Add keywords
\par\vspace{0.3cm}
\textbf{Keywords}: \@keywordsEN{}
\fi



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Acknowledgements
%%
%%%%%

\iftbsi@firstyr
% First and second yr and tecreport don't need the acknowledgements
\else
{\iftbsi@secondyr
%
\else
{\iftbsi@techreport
% removed acknowledgements from techreport
\else
\chapter*{Acknowledgements}
\thispagestyle{empty}
\@acknowledgements{}
\fi}
\fi}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Table of contents, figures, symbols and glossary.
%%
%%%%%

% The following command prevents the page number to be displayed on the first
% page of the TOC.
\addtocontents{toc}{\protect\thispagestyle{empty}}
\pagestyle{empty}
\tableofcontents{}

\iftbsi@glossary%
    \tbsi@printthesisglossary
\fi

\cleardoublepage
\pagestyle{plain}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Backrefs
%%
%%%%%

\iftbsi@backrefs
\RequirePackage[hyperpageref]{backref}
\renewcommand*{\backref}[1]{}
\renewcommand*{\backrefalt}[4]{%
	\ifcase #1 %
	\or
	{\footnotesize Cited on page #2.}%
	\else
	{\footnotesize Cited on pages #2.}%
	\fi
}
\fi


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Presentations
%%
%%%%%

\iftbsi@presentation
%\usetheme{Berkeley}               % TODO: create template 'TBSI'
   % Declare the logo with 1.5cm width
  %\logo{\includegraphics[width=1.5cm]{\collegeshield}}
\fi

%%%%% EOF: tbsi-thesis.cls
