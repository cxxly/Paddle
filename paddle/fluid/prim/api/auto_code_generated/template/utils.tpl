{% macro static_prim_api(op) %}
{{static_prim_api_sig(op)}} {
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("{{op.op_name}}");
  {% filter indent(2, True) %}
    {% for input in op.inputs %}
{{static_prim_api_input(input)}}
    {% endfor %}
    {% for output in op.outputs %}
{{static_prim_api_output(output)}}
    {% endfor %}
    {% for attr in op.attrs %}
{{static_prim_api_attr(attr)}}
    {% endfor %}
  {% endfilter %}
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block); 
  {% if op.outputs|length > 1 %}
  return std::make_tuple({%- for o in op.outputs -%}{{o.name}}{{", " if not loop.last else "" }}{%- endfor -%});
  {% else %}
  return {{op.outputs[0].name}};
  {% endif %}
}
{% endmacro %}


{% macro static_prim_api_sig(op) -%}
template <>
{{static_prim_api_sig_ret(op)}} {{op["name"]}}<DescTensor>({{static_prim_api_sig_params(op)}})
{%- endmacro %}


{%- macro static_prim_api_sig_params(op) -%}
  {%- for input in op.inputs -%}
{{input["typename"] | to_paddle_input_type(input.optional)}} {{input["name"]}}{{", " if not loop.last else "" }}
  {%- endfor -%}
  {%- if op["attrs"]|length > 0 -%} {{", "}} {%- endif -%}
    {%- for attr in op["attrs"] -%}
{{attr["typename"] | to_paddle_attr_type}} {{attr["name"]}}{{ ", " if not loop.last else "" }}
  {%- endfor -%}
{%- endmacro -%}


{%- macro static_prim_api_sig_ret(op) -%}
  {%- if op.outputs|length > 1 -%} 
std::tuple<{%- for o in op.outputs -%}{{o.typename|to_paddle_output_type}}{{", " if not loop.last else "" }}{%- endfor -%}>
  {%- else -%}
{{op.outputs[0].typename | to_paddle_output_type}}
  {%- endif -%}
{%- endmacro -%}


{% macro static_prim_api_input(input) %}
  {%- if input.optional -%}
{{static_prim_api_input_optional(input)}}
  {%- else -%}
{{static_prim_api_input_ignore_optional(input)}}
  {%- endif -%}
{%- endmacro -%}


{%- macro static_prim_api_input_optional(input) -%}
  {%- if input.typename=='Tensor[]' -%}
if ({{input.name}}) {
  std::vector<std::string> {{input.name}}_names;
  std::transform({{input.name}}.get().begin(), {{input.name}}.get().end(), {{input.name}}_names.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<prim::DescTensor>(t.impl())->Name();
  });
  op->SetInput("{{input.fluid_name | to_pascal}}", {{input.name}}_names);  
}
  {%- else -%}
if ({{input.name}}) {
  op->SetInput("{{input.fluid_name | to_pascal}}", {std::static_pointer_cast<prim::DescTensor>({{input.name}}->impl())->Name()});
}
  {%- endif -%}
{%- endmacro -%}


{%- macro static_prim_api_input_ignore_optional(input) -%}
  {%- if input.typename=='Tensor[]' -%}
std::vector<std::string> {{input.name}}_names;
std::transform({{input.name}}.begin(), {{input.name}}.end(), {{input.name}}_names.begin(), [](const Tensor& t) {
  return std::static_pointer_cast<prim::DescTensor>(t.impl())->Name();
});
op->SetInput("{{input.fluid_name | to_pascal}}", {{input.name}}_names);  
  {%- else -%}
op->SetInput("{{input.fluid_name | to_pascal}}", {std::static_pointer_cast<prim::DescTensor>({{input.name}}->impl())->Name()});
  {%- endif -%}
{%- endmacro -%}

{%- macro static_prim_api_output(output) -%}
  {% if- output.intermediate -%}
    {{ignore_intermediate_output()}}
  {%- elif output.typename=='Tensor[]'-%}
std::vector<Tensor> {{output.name}};
std::vector<std::string> {{output.name}}_names;
for (auto i=0; i<{{output.size}}; i++) {
  auto tmp = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
  {{output.name}}.push_back(tmp);
  {{output.name}}_names.push_back(std::static_pointer_cast<prim::DescTensor>(tmp.impl())->Name());
}
op->SetOutput("{{output.fluid_name | to_pascal}}", {{output.name}}_names);
  {%- else -%}
auto {{output.name}} = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
op->SetOutput("{{output.fluid_name | to_pascal}}", {std::static_pointer_cast<prim::DescTensor>({{output.name}}.impl())->Name()});
  {%- endif -%}
{%- endmacro -%}

{% macro static_prim_api_attr(attr) %}
op->SetAttr("{{attr.fluid_name}}", {{phi_attr_to_fluid(attr)}});
{%- endmacro %}


{%- macro ignore_intermediate_output() -%}
{#- Don't redner anything -#}
{%- endmacro -%}


{%- macro phi_attr_to_fluid(attr) -%}
  {%- if attr.typename=='IntArray'-%}
{{int_array_to_fluid(attr.name, attr.typename, attr.fluid_name, attr.data_type)}}
  {%- elif attr.typename=='Scalar'-%}
{{scalar_to_fluid(attr.name, attr.typename, attr.fluid_name, attr.data_type)}}
  {%- elif attr.typename=='DataType'-%}
{{datatype_to_fluid(attr.name, attr.typename, attr.fluid_name, attr.data_type)}}
  {%- else -%}
{{attr.name}}
  {%- endif -%}
{%- endmacro %}


{%- macro int_array_to_fluid(src_name, src_type, dst_name, dst_type) -%}
  {%- if src_type=='IntArray' and dst_type=='std::vector<int>' -%}
unsafe_vector_cast<int64_t, int>({{src_name}}.GetData())
  {%- else -%}
{{src_name}}.GetData()
  {%- endif -%}
{%- endmacro -%}


{%- macro scalar_to_fluid(src_name, src_type, dst_name, dst_type) -%}
{{src_name}}.to<{{dst_type}}>()
{%- endmacro -%}


{%- macro datatype_to_fluid(src_name, src_type, dst_name, dst_type) -%}
paddle::framework::TransToProtoVarType({{src_name}})
{%- endmacro -%}
