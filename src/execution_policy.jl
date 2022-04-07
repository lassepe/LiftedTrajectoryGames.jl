abstract type AbstractExecutionPolicy end
struct SequentialExecutionPolicy <: AbstractExecutionPolicy end
function map_threadable(f, collection, ::SequentialExecutionPolicy)
    map(f, collection)
end

struct MultiThreadedExecutionPolicy <: AbstractExecutionPolicy end
function map_threadable(f, collection, ::MultiThreadedExecutionPolicy)
    ThreadsX.map(f, collection)
end
