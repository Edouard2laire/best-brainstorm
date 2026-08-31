function inv_proj = be_wavelet_inverse_projection_fast(obj,OPTIONS)
%BE_WAVELET_INVERSE_PROJECTION Compute the inverse projection from box to
%time courses

    nbSmp       = size(obj.ImageGridAmp,2);
    nbSmpTime   = size(obj.data,2) ;

    all_scales  = OPTIONS.automatic.selected_samples(2, :);
    all_transls = OPTIONS.automatic.selected_samples(3, :);

    % Pre-compute one wavelet per scale
    [unique_scales, mother_wavelet, required_space] = prepare_wavelet(nbSmpTime, OPTIONS);

    inv_proj = spalloc(nbSmp, nbSmpTime, required_space);
    for iScale = 1:length(unique_scales)
        
        iBoxes = find(all_scales == unique_scales(iScale));

        scales          = all_scales(iBoxes);
        transls         = all_transls(iBoxes);
        inv_wavelet     = mother_wavelet(iScale, :);
            
        shiffting    = 2.^scales(1);
        shiftAmounts = mod(shiffting * (transls(1) - transls(:)), nbSmpTime);
        
        indices = mod(bsxfun(@plus, (1:nbSmpTime)-1, shiftAmounts), nbSmpTime) + 1;
        inv_proj(iBoxes, :) = inv_wavelet(indices);
    end

    inv_proj    =   inv_proj(:,obj.info_extension.start:obj.info_extension.end);
end


function [unique_scales, mother_wavelet, required_space] = prepare_wavelet(nbSmpTime, OPTIONS)
    
    all_scales  = OPTIONS.automatic.selected_samples(2, :);
    all_transls = OPTIONS.automatic.selected_samples(3, :);

    unique_scales = unique(all_scales);
    
    iBoxesRef = zeros(1, length(unique_scales));
    nBoxes    = zeros(1, length(unique_scales));
    for iScale = 1:length(unique_scales)
        tmp = find(all_scales == unique_scales(iScale));
        iBoxesRef(iScale) = tmp(1);
        nBoxes(iScale)    = length(tmp);
    end
    
    x = 1:length(unique_scales);
    y = nbSmpTime ./ (2.^all_scales(iBoxesRef)) + all_transls(iBoxesRef);
    wav = sparse(x, y, 1, length(unique_scales), nbSmpTime);
    mother_wavelet    =   be_wavelet_inverse( wav, OPTIONS );

    required_space = 0;
    for iScale = 1:length(unique_scales)
        required_space = required_space + nnz(mother_wavelet(iScale, :)) * nBoxes(iScale);
    end

end