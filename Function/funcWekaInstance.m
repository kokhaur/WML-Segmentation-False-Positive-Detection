function instances = funcWekaInstance(m_strName, m_FeatureName, m_FeatureData, m_ClassLabel)


fastVector = weka.core.FastVector();
instances = weka.core.Instances(m_strName,fastVector,size(m_FeatureData,1));

% ADDING ATTRIBUTES
for i = 1:numel(m_FeatureName)
    fastVector.addElement(weka.core.Attribute(m_FeatureName{i}));
end

% ADDING OBSERVATIONS
for i = 1:size(m_FeatureData,1)
    instances.add(weka.core.Instance(1,m_FeatureData(i,:))); % instance weight is 1
end
% ADDING CLASS LABELS
values = weka.core.FastVector();
for iter=1: size(m_ClassLabel,2)
    values.addElement(java.lang.String(m_ClassLabel{iter}));
end

% values.addElement(java.lang.String('Non-lesion'));
% values.addElement(java.lang.String('Lesion'));

classAttribute = weka.core.Attribute('class', values);

% Insert attribute at end of instances and set as class index
instances.insertAttributeAt(classAttribute, instances.numAttributes());
idx = instances.numAttributes() - 1;
instances.setClassIndex(idx);