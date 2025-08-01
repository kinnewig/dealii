#!/usr/bin/perl
## ------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2007 - 2023 by the deal.II authors
##
## This file is part of the deal.II library.
##
## Part of the source code is dual licensed under Apache-2.0 WITH
## LLVM-exception OR LGPL-2.1-or-later. Detailed license information
## governing the source code and code contributions can be found in
## LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
##
## ------------------------------------------------------------------------


# read all lines of input. within the loop, first do some
# easy substitutions that are confined to a single line; later
# do some multi-line substitutions as well

while (<>)
{
    ########################################################
    # Step 1: substitutions within a single line
    ########################################################


    # Make sure we can just write $...$ for formulas, except if
    # the dollar sign was escaped with a backslash (use negative
    # look-behind to detect that case, using (?<!\\) to make sure
    # the preceding character -- if there is any -- is not a backslash):
    s#(?<!\\)\$#\@f\$#g;

    # We don't let doxygen put everything into a namespace
    # dealii. consequently, doxygen can't link references that contain an
    # explicit dealii:: qualification. remove it and replace it by the
    # global scope.
    #
    # Try to detect whether or not ::dealii:: is part of a macro (e.g., Assert)
    # continuation line (line ending in '\'): if it is then replace it with
    # spaces. This keeps the '\'s aligned, e.g., the following snippet (which is
    # part of the definition of Assert)
    #
    # ::dealii::internals::abort(__LINE__,             \ # part of macro
    #                            __PRETTY_FUNCTION__)
    #
    # will be converted to
    #
    #         ::internals::abort(__LINE__,             \ # part of macro
    #                            __PRETTY_FUNCTION__)
    #
    # so the '\' is kept in the same column.
    #
    # Now, as of doxygen 1.5, this still produces the wrong result, but
    # that's not our fault. This is reported here:
    #    https://github.com/doxygen/doxygen/issues/2285
    if (m/^ *(::)?dealii::.*\\$/)
    {
        s/::dealii::(.*)\\$/::\1        \\/g;
        s/dealii::(.*)\\$/\1        \\/g;
    }
    elsif (m/using namespace dealii::/)
    {
        # namespace declarations (see, e.g., step-40) in source code don't work
        # if we cut off the 'dealii::' part, so leave it in that case.
    }
    else
    {
        s/(::)?dealii::/::/g;
    }

    # Replace all occurrences of something like step-xx by
    #    @ref step_xx "step-xx"
    # so that doxygen properly cross references them. Before we had
    # this rule, we actually had to write this sequence out in our
    # documentation. Unfortunately, as a consequence, there are vestiges
    # of this style, so we can't substitute things that look like
    # "step-xx" (with the quotes). We therefore do not substitute
    # if step-xx is preceded or followed by quotation marks, or if
    # the text is explicitly preceded by a backslash for escaping.
    #
    # There are other exceptions:
    # - the scripts in doc/doxygen/tutorial produce files that have
    #   table of contents entries. We don't want these cross-linked
    #   to itself.
    # - things like step-12.solution.png that typically appear in
    #   @image commands.
    # - things in headings
    s/(?<![\"\\\/])step-(\d\w*)(?!\")/\@ref step_\1 \"step-\1\"/gi
        if !m/(\@page|\<img|\@image|<h\d>)/i;

    # Now that we have substituted things that involve step-xx text
    # that was supposed to be expanded, we no longer need to guard
    # text that shouldn't be expanded by backslashes. So, if we have
    # \step-xx that was explicitly escaped with a backslash, remove the
    # latter.
    s/\\(step-\d\w*)/\1/g;

    # doxygen version 1.7.1 and later have the habit of thinking that
    # everything that starts with "file:" is the beginning of a link,
    # but we occasionally use this in our tutorials in the form
    # "...this functionality is declared in the following header file:",
    # where it leads to a non-functional link. We can avoid the problem
    # by replacing a "file:" at the end of a line with the text
    # "file :", which doxygen doesn't recognize:
    s#file:[ \t]*$#file :#g;

    # handle tutorial DOIs of the form @dealiiTutorialDOI{link,imglink}
    # and @dealiiTutorialDOI{link}
    if (m/(\@dealiiTutorialDOI\{([^\}]+)\})/)
    {
	@args = split(',', $2, 2);
	$doi = @args[0];
	$url = "https://doi.org/" . $doi;
	$imgurl = @args[1] || '';
	$text = "\@note If you use this program as a basis for your own work, please consider citing it in your list of references. The initial version of this work was contributed to the deal.II project by the authors listed in the following citation: ";
	# Note: We need to wrap the html in @htmlonly @endhtmlonly because doxygen
	# will remove the embedded img tag completely otherwise. Don't ask me why.
	$text = $text . "\@htmlonly <a href=\"$url\">";
	if (length($imgurl) > 0)
	{
	    $text = $text . "<img src=\"$imgurl\" alt=\"$doi\"/>";
	}
	else
	{
	    $text = $text . "$doi";
	}
	$text = $text . "</a> \@endhtmlonly";
	s/(\@dealiiTutorialDOI\{([^\}]+)\})/$text/;
    }

    # Handle commands such as @dealiiVideoLecture{20.5,33} by expanding it
    # into a note with some text
    if (m/(\@dealiiVideoLecture\{([0-9\.]+)((, *[0-9\.]+ *)*)\})/)
    {
        $substext = $1;

        $text = "\@note The material presented here is also discussed in ";

        # add links to the individual lectures
        $text = $text . "<a href=\"https://www.math.colostate.edu/~bangerth/videos.676.$2.html\">video lecture $2</a>";
        
        if (length($3) > 0)
        {
            # if it is a list of lectures, also list the others.
            $x = $3;
            $x =~ s/^, *//g;
            @otherlectures = split (',', "$x");

            foreach $lecture (@otherlectures)
            {
                $text = $text . ", <a href=\"https://www.math.colostate.edu/~bangerth/videos.676.$lecture.html\">video lecture $lecture</a>";
            }
        }

        $text = $text . ". (All video lectures are also available <a href=\"https://www.math.colostate.edu/~bangerth/videos.html\">here</a>.)";
        s/(\@dealiiVideoLecture\{([0-9\.]+)((, *[0-9\.]+ *)*)\})/$text/;
    }


    # @dealiiVideoLectureSeeAlso works as above, but just expands into
    # regular text, no @note
    if (m/(\@dealiiVideoLectureSeeAlso\{([0-9\.]+)((, *[0-9\.]+ *)*)\})/)
    {
        $substext = $1;

        $text = "See also ";

        # add links to the individual lectures
        $text = $text . "<a href=\"https://www.math.colostate.edu/~bangerth/videos.676.$2.html\">video lecture $2</a>";
        
        if (length($3) > 0)
        {
            $x = $3;
            $x =~ s/^, *//g;
            @otherlectures = split (',', "$x");

            foreach $lecture (@otherlectures)
            {
                $text = $text . ", <a href=\"https://www.math.colostate.edu/~bangerth/videos.676.$lecture.html\">video lecture $lecture</a>";
            }
        }

        $text = $text . ".";
        s/(\@dealiiVideoLectureSeeAlso\{([0-9\.]+)((, *[0-9\.]+ *)*)\})/$text/;
    }



    ########################################################
    # Step 2: substitutions for patterns that span
    #         multiple lines
    ########################################################

    # NOTE: The doxygen documentation says that "the filter must not
    # add or remove lines; it is applied before the code is scanned,
    # but not when the output code is generated. If lines are added or
    # removed, the anchors will not be placed correctly."
    #
    # In other words, filters below can munch multiple lines, but they
    # must print the result on as many lines as they were on before.


    # Finally output the last line of what has been substituted
    print;
}
