
# To build the documentation:
#    - julia --project="." make.jl
#    - empty!(ARGS); include("make.jl")
# To build the documentation without running the tests:
#    - julia --project="." make.jl preview
#    - push!(ARGS,"preview"); include("make.jl")

# !!! note "An optional title"
#    4 spaces idented
# note, tip, warning, danger, compat



# Format notes:


# # A markdown H1 title
# A non-code markdown normal line

## A comment within the code chunk

#src: line exclusive to the source code and thus filtered out unconditionally

using Pkg
cd(@__DIR__)
Pkg.activate(".")

#Pkg.resolve()
Pkg.instantiate()
#Pkg.add(["Documenter", "Literate", "Glob", "DataFrames", "OdsIO"])

using Test, Documenter, DocumenterMarkdown, Glob


const PAGES_ROOTDIR = joinpath(@__DIR__, "srcPages")
# Important: If some lesson is removed but the md file is left, this may still be used by Documenter

const PAGES_ROOTDIR_TMP = joinpath(@__DIR__, "srcPages_tmp")
# Where to save the PAGES before they are preprocessed

MAKE_PDF = false


# Utility functions.....

function link_example(content)
    edit_url = match(r"EditURL = \"(.+?)\"", content)[1]
    footer = match(r"^(---\n\n\*This page was generated using)"m, content)[1]
    content = replace(
        content, footer => "[View this file on Github]($(edit_url)).\n\n" * footer
    )
    return content
end


"""
    include_sandbox(filename)
Include the `filename` in a temporary module that acts as a sandbox. (Ensuring
no constants or functions leak into other files.)
"""
function include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

function makeList(dir)
    outArray = Pair{String,String}[]
    for file in filter(file -> endswith(file, ".md"), sort(readdir(dir)))
        if file == "index.md"
            continue
        end
        displayFilename = replace(file,".md"=>"")
        displayFilename = replace(displayFilename,"_"=>" ")[4:end]
        push!(outArray,displayFilename=>file)
    end
    return outArray
end


makeList(PAGES_ROOTDIR)


"""
    rdir(string,match)

Return a vector of all files (full paths) of a given directory recursivelly that matches `match`, recursivelly.

# example
filenames = getindex.(splitdir.(rdir(PAGES_ROOTDIR,"*.jl")),2) #get all md filenames
"""
function rdir(dir::AbstractString, pat::Glob.FilenameMatch)
    result = String[]
    for (root, dirs, files) in walkdir(dir)
        append!(result, filter!(f -> occursin(pat, f), joinpath.(root, files)))
    end
    return result
end
rdir(dir::AbstractString, pat::AbstractString) = rdir(dir, Glob.FilenameMatch(pat))


function preprocess(rootDir)
    cd(@__DIR__)
    Pkg.activate(".")
    #rootDir = PAGES_ROOTDIR 
    files  = rdir(rootDir,"*.md")
    for file in files
        #file = files[4]
        origContent = read(file,String)
        outContent = ""
        filename = splitdir(file)[2]
        outContent *= origContent
        outContent = replace(outContent, "```julia" => "```@example gtnotes")
        if (filename != "01_index.md")
            commentCode = """
                ```@raw html
                <script src="https://utteranc.es/client.js"
                        repo="sylvaticus/GameTheoryNotes"
                        issue-term="title"
                        label="💬 website_comment"
                        theme="github-dark"
                        crossorigin="anonymous"
                        async>
                </script>
                ```
                """
            addThisCode1 = """
                ```@raw html
                <div class="addthis_inline_share_toolbox"></div>
                ```
                """
            addThisCode2 = """
                ```@raw html
                <!-- Go to www.addthis.com/dashboard to customize your tools -->
                <script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-6256c971c4f745bc"></script>
                ```
                """
            # https://crowdsignal.com/support/rating-widget/
            ratingCode1 = """
                ```@raw html
                <div id="pd_rating_holder_8962705"></div>
                <script type="text/javascript">
                const pageURL = window.location.href;
                PDRTJS_settings_8962705 = {
                "id" : "8962705",
                "unique_id" : "$(file)",
                "title" : "$(filename)",
                "permalink" : pageURL
                };
                </script>
                ```
                """
            ratingCode2 = """
                ```@raw html
                <script type="text/javascript" charset="utf-8" src="https://polldaddy.com/js/rating/rating.js"></script>
                ```
                """
            outContent *= "\n---------\n"
            outContent *= ratingCode1
            outContent *= addThisCode1
            outContent *= "\n---------\n"
            outContent *= commentCode
            outContent *= ratingCode2
            outContent *= addThisCode2
        end
        write(file,outContent)
    end # end for each file
end #end preprocess function

# ------------------------------------------------------------------------------
# Saving the unmodified source to a temp directory
cp(PAGES_ROOTDIR, PAGES_ROOTDIR_TMP; force=true)



if MAKE_PDF
    println("Starting making PDF...")
    makedocs(sitename="Game Theory Notes",
            authors = "Antonello Lobianco",
            pages = [
                Index => "index.md",
                "API" => makeList(PAGES_ROOTDIR)
            ],
            format = Markdown(),
            source  = "srcPages", # Attention here !!!!!!!!!!!
            build   = "buildedPages_PDF",
    )
end



println("Starting preprocessing markdown pages...")
preprocess(PAGES_ROOTDIR)

println("Starting making the documentation...")
makedocs(sitename="GameTheoryNotes",
         authors = "Antonello Lobianco",
         pages = [
            "Index" => "index.md",
            "Notes" => makeList(PAGES_ROOTDIR)
        ],
         format = Documenter.HTML(
             prettyurls = false,
             analytics = "G-CNCXWWMQ38",
             assets = ["assets/custom.css"],
             ),
         #strict = true,
         doctest = true,
         source  = "srcPages", # Attention here !!!!!!!!!!!
         build   = "buildedPages",
         #preprocess = preprocess
)





# Copying back the unmodified source
cp(PAGES_ROOTDIR_TMP, PAGES_ROOTDIR; force=true)

println("Starting deploying the documentation...")
deploydocs(
    repo = "github.com/sylvaticus/GameTheoryNotes.git",
    devbranch = "main",
    target = "buildedPages"
)
